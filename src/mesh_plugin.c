/*
 * NCCL Mesh Plugin - Main Implementation
 * 
 * Subnet-aware RDMA transport for direct-connect mesh topologies
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <dirent.h>
#include <pthread.h>

#include <infiniband/verbs.h>

#include "nccl/net.h"
#include "mesh_plugin.h"

// Global state
struct mesh_plugin_state g_mesh_state = {0};

// Plugin name
#define PLUGIN_NAME "Mesh"

/*
 * Utility: Convert IP string to uint32
 */
uint32_t mesh_ip_to_uint(const char *ip_str) {
    struct in_addr addr;
    if (inet_pton(AF_INET, ip_str, &addr) != 1) {
        return 0;
    }
    return ntohl(addr.s_addr);
}

/*
 * Utility: Convert uint32 to IP string
 */
void mesh_uint_to_ip(uint32_t ip, char *buf, size_t len) {
    struct in_addr addr;
    addr.s_addr = htonl(ip);
    inet_ntop(AF_INET, &addr, buf, len);
}

/*
 * Get IP address and netmask for a network interface
 */
int mesh_get_interface_ip(const char *if_name, uint32_t *ip, uint32_t *mask) {
    struct ifaddrs *ifaddr, *ifa;
    int found = 0;
    
    if (getifaddrs(&ifaddr) == -1) {
        return -1;
    }
    
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL) continue;
        if (ifa->ifa_addr->sa_family != AF_INET) continue;
        if (strcmp(ifa->ifa_name, if_name) != 0) continue;
        
        struct sockaddr_in *addr = (struct sockaddr_in *)ifa->ifa_addr;
        struct sockaddr_in *netmask = (struct sockaddr_in *)ifa->ifa_netmask;
        
        *ip = ntohl(addr->sin_addr.s_addr);
        *mask = ntohl(netmask->sin_addr.s_addr);
        found = 1;
        break;
    }
    
    freeifaddrs(ifaddr);
    return found ? 0 : -1;
}

/*
 * Find network interface name for an RDMA device
 * Looks in /sys/class/infiniband/<rdma_dev>/device/net/
 */
const char* mesh_find_netdev_for_rdma(const char *rdma_dev) {
    static char netdev[64];
    char path[256];
    DIR *dir;
    struct dirent *entry;
    
    snprintf(path, sizeof(path), "/sys/class/infiniband/%s/device/net", rdma_dev);
    dir = opendir(path);
    if (!dir) {
        return NULL;
    }
    
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] != '.') {
            strncpy(netdev, entry->d_name, sizeof(netdev) - 1);
            netdev[sizeof(netdev) - 1] = '\0';
            closedir(dir);
            return netdev;
        }
    }
    
    closedir(dir);
    return NULL;
}

/*
 * Find the NIC that can reach a given IP address (same subnet)
 */
struct mesh_nic* mesh_find_nic_for_ip(uint32_t peer_ip) {
    for (int i = 0; i < g_mesh_state.num_nics; i++) {
        struct mesh_nic *nic = &g_mesh_state.nics[i];
        uint32_t peer_subnet = peer_ip & nic->netmask;
        
        MESH_DEBUG("Checking NIC %s: peer_ip=0x%x, subnet=0x%x, nic_subnet=0x%x",
                   nic->dev_name, peer_ip, peer_subnet, nic->subnet);
        
        if (peer_subnet == nic->subnet) {
            MESH_DEBUG("Found matching NIC %s for peer IP 0x%x", 
                       nic->dev_name, peer_ip);
            return nic;
        }
    }
    
    MESH_WARN("No NIC found for peer IP 0x%x", peer_ip);
    return NULL;
}

/*
 * Setup a single NIC
 */
int mesh_setup_nic(struct mesh_nic *nic, struct ibv_device *device) {
    struct ibv_device_attr dev_attr;
    struct ibv_port_attr port_attr;
    
    // Open context
    nic->context = ibv_open_device(device);
    if (!nic->context) {
        MESH_WARN("Failed to open device %s", ibv_get_device_name(device));
        return -1;
    }
    
    // Get device name
    strncpy(nic->dev_name, ibv_get_device_name(device), sizeof(nic->dev_name) - 1);
    
    // Find associated network interface
    const char *netdev = mesh_find_netdev_for_rdma(nic->dev_name);
    if (netdev) {
        strncpy(nic->if_name, netdev, sizeof(nic->if_name) - 1);
        
        // Get IP address
        if (mesh_get_interface_ip(nic->if_name, &nic->ip_addr, &nic->netmask) == 0) {
            nic->subnet = nic->ip_addr & nic->netmask;
            
            char ip_str[INET_ADDRSTRLEN], mask_str[INET_ADDRSTRLEN], subnet_str[INET_ADDRSTRLEN];
            mesh_uint_to_ip(nic->ip_addr, ip_str, sizeof(ip_str));
            mesh_uint_to_ip(nic->netmask, mask_str, sizeof(mask_str));
            mesh_uint_to_ip(nic->subnet, subnet_str, sizeof(subnet_str));
            
            MESH_INFO("NIC %s (%s): IP=%s, mask=%s, subnet=%s",
                      nic->dev_name, nic->if_name, ip_str, mask_str, subnet_str);
        } else {
            MESH_WARN("Could not get IP for interface %s", nic->if_name);
        }
    } else {
        MESH_WARN("Could not find netdev for RDMA device %s", nic->dev_name);
    }
    
    // Query device attributes
    if (ibv_query_device(nic->context, &dev_attr)) {
        MESH_WARN("Failed to query device %s", nic->dev_name);
        ibv_close_device(nic->context);
        return -1;
    }
    
    nic->max_qp = dev_attr.max_qp;
    nic->max_cq = dev_attr.max_cq;
    nic->max_mr = dev_attr.max_mr;
    nic->max_sge = dev_attr.max_sge;
    nic->max_mr_size = dev_attr.max_mr_size;
    
    // Query port (assume port 1)
    nic->port_num = 1;
    if (ibv_query_port(nic->context, nic->port_num, &port_attr)) {
        MESH_WARN("Failed to query port for %s", nic->dev_name);
        ibv_close_device(nic->context);
        return -1;
    }

    // Store the port's active MTU (TICKET-9: avoid MTU mismatch on RoCE)
    nic->active_mtu = port_attr.active_mtu;
    MESH_DEBUG("NIC %s port %d: active_mtu=%d", nic->dev_name, nic->port_num, nic->active_mtu);

    // Allocate protection domain
    nic->pd = ibv_alloc_pd(nic->context);
    if (!nic->pd) {
        MESH_WARN("Failed to allocate PD for %s", nic->dev_name);
        ibv_close_device(nic->context);
        return -1;
    }
    
    // Use configured GID index or default to 3 (RoCE v2 with IPv4)
    nic->gid_index = g_mesh_state.gid_index;

    // Cache the GID at initialization (TICKET-5)
    if (mesh_cache_gid(nic) != 0) {
        MESH_WARN("Failed to cache GID for %s, will query on demand", nic->dev_name);
    }

    MESH_INFO("Initialized NIC %s: max_qp=%d, max_mr=%d, gid_index=%d",
              nic->dev_name, nic->max_qp, nic->max_mr, nic->gid_index);

    return 0;
}

/*
 * Cache the GID for a NIC (TICKET-5)
 * Called during initialization to avoid repeated queries
 */
int mesh_cache_gid(struct mesh_nic *nic) {
    union ibv_gid gid;
    struct timespec ts;

    if (!nic || !nic->context) {
        return -1;
    }

    if (ibv_query_gid(nic->context, nic->port_num, nic->gid_index, &gid) != 0) {
        MESH_WARN("Failed to query GID for %s port %d index %d: %s",
                  nic->dev_name, nic->port_num, nic->gid_index, strerror(errno));
        nic->gid_valid = 0;
        return -1;
    }

    memcpy(&nic->cached_gid, &gid, sizeof(gid));
    nic->gid_valid = 1;

    // Record query time for staleness detection
    clock_gettime(CLOCK_MONOTONIC, &ts);
    nic->gid_query_time = ts.tv_sec;

    MESH_DEBUG("Cached GID for %s: %02x%02x:%02x%02x:%02x%02x:%02x%02x:...",
               nic->dev_name,
               gid.raw[0], gid.raw[1], gid.raw[2], gid.raw[3],
               gid.raw[4], gid.raw[5], gid.raw[6], gid.raw[7]);

    return 0;
}

/*
 * Validate that the cached GID is still current (TICKET-5)
 * Returns 0 if valid, -1 if changed (and logs warning)
 * This handles GID table changes gracefully rather than flooding warnings
 */
int mesh_validate_gid(struct mesh_nic *nic) {
    union ibv_gid current_gid;
    struct timespec ts;
    uint64_t now;

    if (!nic || !nic->context) {
        return -1;
    }

    // If GID was never cached, try to cache it now
    if (!nic->gid_valid) {
        return mesh_cache_gid(nic);
    }

    // Check staleness - only re-validate every 60 seconds to avoid excessive queries
    clock_gettime(CLOCK_MONOTONIC, &ts);
    now = ts.tv_sec;
    if (now - nic->gid_query_time < 60) {
        return 0;  // Cached value is fresh enough
    }

    // Query current GID
    if (ibv_query_gid(nic->context, nic->port_num, nic->gid_index, &current_gid) != 0) {
        MESH_WARN("GID validation query failed for %s: %s", nic->dev_name, strerror(errno));
        return -1;
    }

    // Compare with cached value
    if (memcmp(&current_gid, &nic->cached_gid, sizeof(current_gid)) != 0) {
        // GID changed - log once and update cache
        MESH_WARN("GID table changed for %s (index %d), updating cache",
                  nic->dev_name, nic->gid_index);
        memcpy(&nic->cached_gid, &current_gid, sizeof(current_gid));
        nic->gid_query_time = now;
        // Return success - we've handled the change gracefully
        return 0;
    }

    // GID unchanged, update timestamp
    nic->gid_query_time = now;
    return 0;
}

/*
 * Get GID for a NIC, using cached value when possible (TICKET-5)
 * This is the main entry point for getting GID - avoids repeated queries
 */
int mesh_get_gid(struct mesh_nic *nic, union ibv_gid *gid) {
    if (!nic || !gid) {
        return -1;
    }

    // Ensure cache is valid
    if (mesh_validate_gid(nic) != 0) {
        return -1;
    }

    memcpy(gid, &nic->cached_gid, sizeof(*gid));
    return 0;
}

/*
 * Initialize all NICs
 */
int mesh_init_nics(void) {
    struct ibv_device **dev_list;
    int num_devices;
    
    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) {
        MESH_WARN("Failed to get RDMA device list");
        return -1;
    }
    
    if (num_devices == 0) {
        MESH_WARN("No RDMA devices found");
        ibv_free_device_list(dev_list);
        return -1;
    }
    
    MESH_INFO("Found %d RDMA devices", num_devices);
    
    g_mesh_state.num_nics = 0;
    for (int i = 0; i < num_devices && g_mesh_state.num_nics < MESH_MAX_NICS; i++) {
        struct mesh_nic *nic = &g_mesh_state.nics[g_mesh_state.num_nics];
        memset(nic, 0, sizeof(*nic));
        
        if (mesh_setup_nic(nic, dev_list[i]) == 0) {
            // Only count NICs that have an IP configured
            if (nic->ip_addr != 0) {
                g_mesh_state.num_nics++;
            } else {
                // Clean up NIC without IP
                if (nic->pd) ibv_dealloc_pd(nic->pd);
                if (nic->context) ibv_close_device(nic->context);
            }
        }
    }
    
    ibv_free_device_list(dev_list);
    
    MESH_INFO("Initialized %d NICs with IP addresses", g_mesh_state.num_nics);
    return g_mesh_state.num_nics > 0 ? 0 : -1;
}

/*
 * Create a listening socket for QP handshake
 */
int mesh_create_handshake_socket(uint32_t bind_ip, uint16_t *port_out) {
    int sock;
    struct sockaddr_in addr;
    socklen_t addrlen = sizeof(addr);
    int opt = 1;
    
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        MESH_WARN("Failed to create handshake socket: %s", strerror(errno));
        return -1;
    }
    
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(bind_ip);
    addr.sin_port = 0;  // Let OS choose port
    
    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        MESH_WARN("Failed to bind handshake socket: %s", strerror(errno));
        close(sock);
        return -1;
    }
    
    if (listen(sock, 16) < 0) {
        MESH_WARN("Failed to listen on handshake socket: %s", strerror(errno));
        close(sock);
        return -1;
    }
    
    // Get assigned port
    if (getsockname(sock, (struct sockaddr *)&addr, &addrlen) < 0) {
        MESH_WARN("Failed to get socket name: %s", strerror(errno));
        close(sock);
        return -1;
    }
    
    *port_out = ntohs(addr.sin_port);
    
    char ip_str[INET_ADDRSTRLEN];
    mesh_uint_to_ip(bind_ip, ip_str, sizeof(ip_str));
    MESH_INFO("Handshake socket listening on %s:%d", ip_str, *port_out);
    
    return sock;
}

/*
 * Accept a handshake connection, receive remote QP info, and send ours back
 */
int mesh_accept_handshake(int listen_sock, struct mesh_qp_info *remote_info, struct mesh_qp_info *local_info) {
    int conn_sock;
    struct sockaddr_in addr;
    socklen_t addrlen = sizeof(addr);
    
    conn_sock = accept(listen_sock, (struct sockaddr *)&addr, &addrlen);
    if (conn_sock < 0) {
        MESH_WARN("Failed to accept handshake connection: %s", strerror(errno));
        return -1;
    }
    
    char ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &addr.sin_addr, ip_str, sizeof(ip_str));
    MESH_INFO("Accepted handshake connection from %s:%d", ip_str, ntohs(addr.sin_port));
    
    // Receive remote QP info
    ssize_t n = recv(conn_sock, remote_info, sizeof(*remote_info), MSG_WAITALL);
    if (n != sizeof(*remote_info)) {
        MESH_WARN("Failed to receive QP info: got %zd bytes, expected %zu", n, sizeof(*remote_info));
        close(conn_sock);
        return -1;
    }
    
    MESH_INFO("Received remote QP info: qp_num=%u, psn=%u", 
              ntohl(remote_info->qp_num), ntohl(remote_info->psn));
    
    // Send our QP info back
    n = send(conn_sock, local_info, sizeof(*local_info), 0);
    if (n != sizeof(*local_info)) {
        MESH_WARN("Failed to send local QP info: sent %zd bytes, expected %zu", n, sizeof(*local_info));
        close(conn_sock);
        return -1;
    }
    
    MESH_INFO("Sent local QP info: qp_num=%u, psn=%u",
              ntohl(local_info->qp_num), ntohl(local_info->psn));
    
    close(conn_sock);
    return 0;
}

/*
 * Connect, send our QP info, and receive remote's QP info
 * Uses non-blocking connect with select() to avoid deadlock
 * Retry timing controlled by NCCL_MESH_TIMEOUT_MS and NCCL_MESH_RETRY_COUNT
 */
int mesh_send_handshake(uint32_t remote_ip, uint16_t remote_port,
                        struct mesh_qp_info *local_info, struct mesh_qp_info *remote_info) {
    int sock;
    struct sockaddr_in addr;
    int connected = 0;

    // Calculate retry parameters from config
    // timeout_ms spread across retry_count attempts, with 100ms per attempt
    int retry_interval_ms = 100;
    int max_retries = g_mesh_state.timeout_ms / retry_interval_ms;
    if (max_retries < g_mesh_state.retry_count) max_retries = g_mesh_state.retry_count;
    int retries = max_retries;

    char ip_str[INET_ADDRSTRLEN];
    mesh_uint_to_ip(remote_ip, ip_str, sizeof(ip_str));
    MESH_DEBUG("send_handshake: connecting to %s:%d (max_retries=%d)", ip_str, remote_port, max_retries);

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(remote_ip);
    addr.sin_port = htons(remote_port);

    // Retry connection - peer's accept() might not be ready yet
    while (retries > 0 && !connected) {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            MESH_WARN("Failed to create handshake socket: %s", strerror(errno));
            return -1;
        }

        // Set non-blocking
        int flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, flags | O_NONBLOCK);

        int ret = connect(sock, (struct sockaddr *)&addr, sizeof(addr));
        if (ret == 0) {
            connected = 1;
        } else if (errno == EINPROGRESS) {
            // Wait for connection with select
            fd_set writefds;
            struct timeval tv;
            FD_ZERO(&writefds);
            FD_SET(sock, &writefds);
            tv.tv_sec = 0;
            tv.tv_usec = retry_interval_ms * 1000;

            ret = select(sock + 1, NULL, &writefds, NULL, &tv);
            if (ret > 0) {
                // Check if connection succeeded
                int error = 0;
                socklen_t len = sizeof(error);
                getsockopt(sock, SOL_SOCKET, SO_ERROR, &error, &len);
                if (error == 0) {
                    connected = 1;
                } else {
                    close(sock);
                    retries--;
                }
            } else {
                close(sock);
                retries--;
            }
        } else {
            close(sock);
            retries--;
            usleep(retry_interval_ms * 1000);
        }
    }

    if (!connected) {
        MESH_WARN("Failed to connect handshake socket to %s:%d after %d retries", ip_str, remote_port, max_retries);
        return -1;
    }
    
    // Set back to blocking for send/recv
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags & ~O_NONBLOCK);
    
    
    // Send our QP info
    ssize_t n = send(sock, local_info, sizeof(*local_info), 0);
    if (n != sizeof(*local_info)) {
        MESH_WARN("Failed to send QP info: sent %zd bytes, expected %zu", n, sizeof(*local_info));
        close(sock);
        return -1;
    }
    
    
    // Receive remote's QP info (the accept side's NEW QP)
    n = recv(sock, remote_info, sizeof(*remote_info), MSG_WAITALL);
    if (n != sizeof(*remote_info)) {
        MESH_WARN("Failed to receive remote QP info: got %zd bytes, expected %zu", n, sizeof(*remote_info));
        close(sock);
        return -1;
    }
    
    
    close(sock);
    return 0;
}

/*
 * Background handshake thread
 * Handles incoming TCP connections, creates QPs, and queues for accept()
 */
static void *handshake_thread_func(void *arg) {
    struct mesh_listen_comm *lcomm = (struct mesh_listen_comm *)arg;
    
    
    // Set socket to non-blocking so we can check stop flag
    int flags = fcntl(lcomm->handshake_sock, F_GETFL, 0);
    fcntl(lcomm->handshake_sock, F_SETFL, flags | O_NONBLOCK);
    
    while (!lcomm->thread_stop) {
        struct sockaddr_in addr;
        socklen_t addrlen = sizeof(addr);
        
        int conn_sock = accept(lcomm->handshake_sock, (struct sockaddr *)&addr, &addrlen);
        if (conn_sock < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                usleep(10000);  // 10ms
                continue;
            }
            break;
        }
        
        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &addr.sin_addr, ip_str, sizeof(ip_str));
        
        // Receive remote QP info
        struct mesh_qp_info remote_info;
        ssize_t n = recv(conn_sock, &remote_info, sizeof(remote_info), MSG_WAITALL);
        if (n != sizeof(remote_info)) {
            close(conn_sock);
            continue;
        }
        
        MESH_DEBUG("Handshake thread: received QP %u, nic_idx=%d", ntohl(remote_info.qp_num), remote_info.nic_idx);

        // Select NIC based on nic_idx from remote
        int nic_idx = remote_info.nic_idx;
        if (nic_idx >= lcomm->num_qps) nic_idx = 0;
        struct mesh_nic *nic = lcomm->qps[nic_idx].nic;
        
        // Create new QP for this connection
        struct ibv_qp *new_qp = NULL;
        struct ibv_cq *new_cq = NULL;
        if (mesh_create_qp(nic, &new_qp, &new_cq) != 0) {
            close(conn_sock);
            continue;
        }
        
        MESH_DEBUG("Handshake thread: created QP %d on %s", new_qp->qp_num, nic->dev_name);

        // Connect our QP to remote's QP
        struct mesh_handle connect_handle;
        memset(&connect_handle, 0, sizeof(connect_handle));
        connect_handle.qp_num = ntohl(remote_info.qp_num);
        connect_handle.psn = ntohl(remote_info.psn);
        connect_handle.port_num = nic->port_num;
        connect_handle.mtu = nic->active_mtu;  // Use NIC's actual MTU (TICKET-9)

        // Construct GID from remote IP
        union ibv_gid remote_gid;
        memset(&remote_gid, 0, sizeof(remote_gid));
        remote_gid.raw[10] = 0xff;
        remote_gid.raw[11] = 0xff;
        uint32_t remote_ip = remote_info.ip;
        memcpy(&remote_gid.raw[12], &remote_ip, 4);
        connect_handle.gid = remote_gid;
        
        if (mesh_connect_qp(new_qp, nic, &connect_handle) != 0) {
            ibv_destroy_qp(new_qp);
            ibv_destroy_cq(new_cq);
            close(conn_sock);
            continue;
        }
        
        MESH_DEBUG("Handshake thread: QP connected to remote QP %d", connect_handle.qp_num);

        // Send our QP info back
        struct mesh_qp_info local_info;
        memset(&local_info, 0, sizeof(local_info));
        local_info.qp_num = htonl(new_qp->qp_num);
        local_info.psn = htonl(0);
        local_info.ip = htonl(nic->ip_addr);
        local_info.nic_idx = nic_idx;
        
        n = send(conn_sock, &local_info, sizeof(local_info), 0);
        close(conn_sock);
        
        if (n != sizeof(local_info)) {
            ibv_destroy_qp(new_qp);
            ibv_destroy_cq(new_cq);
            continue;
        }
        
        MESH_DEBUG("Handshake thread: sent QP %d back, queueing for accept", new_qp->qp_num);

        // Queue this handshake for accept() to consume
        pthread_mutex_lock(&lcomm->queue_mutex);
        int next_tail = (lcomm->queue_tail + 1) % HANDSHAKE_QUEUE_SIZE;
        if (next_tail != lcomm->queue_head) {
            struct handshake_entry *entry = &lcomm->handshake_queue[lcomm->queue_tail];
            entry->remote_info = remote_info;
            entry->local_qp = new_qp;
            entry->local_cq = new_cq;
            entry->nic = nic;
            entry->valid = 1;
            lcomm->queue_tail = next_tail;
            pthread_cond_signal(&lcomm->queue_cond);
        } else {
            ibv_destroy_qp(new_qp);
            ibv_destroy_cq(new_cq);
        }
        pthread_mutex_unlock(&lcomm->queue_mutex);
    }
    
    return NULL;
}

/*
 * Create QP and CQ on a NIC
 */
int mesh_create_qp(struct mesh_nic *nic, struct ibv_qp **qp_out, struct ibv_cq **cq_out) {
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_qp_init_attr qp_init_attr;
    int err;

    // Create completion queue
    cq = ibv_create_cq(nic->context, 4096, NULL, NULL, 0);
    if (!cq) {
        err = errno;
        MESH_WARN("Failed to create CQ on %s: errno=%d (%s)", nic->dev_name, err, strerror(err));
        return -1;
    }

    // Create queue pair
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.send_cq = cq;
    qp_init_attr.recv_cq = cq;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.cap.max_send_wr = 64;
    qp_init_attr.cap.max_recv_wr = 64;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;
    qp_init_attr.cap.max_inline_data = 64;

    qp = ibv_create_qp(nic->pd, &qp_init_attr);
    if (!qp) {
        err = errno;
        MESH_WARN("Failed to create QP on %s: errno=%d (%s)", nic->dev_name, err, strerror(err));
        ibv_destroy_cq(cq);
        return -1;
    }

    // Transition QP to INIT state
    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = nic->port_num;
    qp_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    if (ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        err = errno;
        MESH_WARN("Failed to transition QP to INIT on %s: errno=%d (%s)", nic->dev_name, err, strerror(err));
        ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        return -1;
    }

    *qp_out = qp;
    *cq_out = cq;
    return 0;
}

/*
 * Connect QP to remote peer with retry logic
 *
 * The RTR transition can fail with ETIMEDOUT (110) if the remote QP isn't ready yet.
 * This happens during rapid collective initialization (FSDP/ZeRO-3) when both sides
 * are racing to connect. We retry with exponential backoff to handle this.
 */
int mesh_connect_qp(struct ibv_qp *qp, struct mesh_nic *nic, struct mesh_handle *remote) {
    struct ibv_qp_attr qp_attr;
    int ret;
    int max_retries = 5;
    int retry_delay_ms = 10;  // Start with 10ms, double each retry

    // Transition to RTR (Ready to Receive)
    // This is where most timeouts occur - remote QP may not be ready yet
    // Use the MTU from connect_handle (caller negotiates min of local/remote)
    enum ibv_mtu path_mtu = remote->mtu ? remote->mtu : nic->active_mtu;
    MESH_DEBUG("QP connect: using path_mtu=%d (remote->mtu=%d, nic->active_mtu=%d)",
               path_mtu, remote->mtu, nic->active_mtu);

    for (int attempt = 0; attempt < max_retries; attempt++) {
        memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.qp_state = IBV_QPS_RTR;
        qp_attr.path_mtu = path_mtu;
        qp_attr.dest_qp_num = remote->qp_num;
        qp_attr.rq_psn = remote->psn;
        qp_attr.max_dest_rd_atomic = 1;
        qp_attr.min_rnr_timer = 12;  // ~0.01ms min RNR NAK timer
        qp_attr.ah_attr.is_global = 1;
        qp_attr.ah_attr.grh.dgid = remote->gid;
        qp_attr.ah_attr.grh.sgid_index = nic->gid_index;
        qp_attr.ah_attr.grh.hop_limit = 64;
        qp_attr.ah_attr.dlid = remote->lid;
        qp_attr.ah_attr.sl = 0;
        qp_attr.ah_attr.src_path_bits = 0;
        qp_attr.ah_attr.port_num = nic->port_num;

        ret = ibv_modify_qp(qp, &qp_attr,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);

        if (ret == 0) {
            break;  // Success
        }

        int err = errno;
        if (attempt < max_retries - 1) {
            MESH_WARN("QP RTR transition failed (attempt %d/%d): errno=%d (%s), retrying in %dms",
                      attempt + 1, max_retries, err, strerror(err), retry_delay_ms);
            usleep(retry_delay_ms * 1000);
            retry_delay_ms *= 2;  // Exponential backoff
        } else {
            MESH_WARN("QP RTR transition failed after %d attempts: errno=%d (%s), dest_qp=%u, gid_index=%d",
                      max_retries, err, strerror(err), remote->qp_num, nic->gid_index);
            return -1;
        }
    }

    // Transition to RTS (Ready to Send)
    // In fast-fail mode: use shorter timeout (14 = ~67ms) and fewer retries (3)
    // Normal mode: longer timeout (18 = ~1s) and max retries (7) for reliability
    int qp_timeout = g_mesh_state.fast_fail ? 14 : 18;
    int qp_retry_cnt = g_mesh_state.fast_fail ? 3 : 7;
    int qp_rnr_retry = g_mesh_state.fast_fail ? 3 : 7;

    for (int attempt = 0; attempt < max_retries; attempt++) {
        memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.qp_state = IBV_QPS_RTS;
        qp_attr.timeout = qp_timeout;
        qp_attr.retry_cnt = qp_retry_cnt;
        qp_attr.rnr_retry = qp_rnr_retry;
        qp_attr.sq_psn = 0;
        qp_attr.max_rd_atomic = 1;

        ret = ibv_modify_qp(qp, &qp_attr,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);

        if (ret == 0) {
            break;  // Success
        }

        int err = errno;
        if (attempt < max_retries - 1) {
            MESH_WARN("QP RTS transition failed (attempt %d/%d): errno=%d (%s), retrying in %dms",
                      attempt + 1, max_retries, err, strerror(err), retry_delay_ms);
            usleep(retry_delay_ms * 1000);
            retry_delay_ms *= 2;
        } else {
            MESH_WARN("QP RTS transition failed after %d attempts: errno=%d (%s)",
                      max_retries, err, strerror(err));
            return -1;
        }
    }

    return 0;
}

/*
 * ============================================================================
 * Connection Pool Implementation (TICKET-6)
 * ============================================================================
 *
 * Pools QP connections for reuse between same node pairs. This significantly
 * improves performance when FSDP/ZeRO creates many communicators during the
 * wrap phase, as we avoid the overhead of creating new QPs for each one.
 */

/*
 * Initialize the connection pool
 */
int mesh_conn_pool_init(void) {
    struct mesh_conn_pool *pool = &g_mesh_state.conn_pool;

    memset(pool, 0, sizeof(*pool));
    pthread_mutex_init(&pool->mutex, NULL);

    MESH_INFO("Connection pool initialized (max %d entries)", MESH_CONN_POOL_SIZE);
    return 0;
}

/*
 * Destroy the connection pool and clean up all connections
 */
void mesh_conn_pool_destroy(void) {
    struct mesh_conn_pool *pool = &g_mesh_state.conn_pool;

    pthread_mutex_lock(&pool->mutex);

    for (int i = 0; i < pool->num_entries; i++) {
        struct mesh_conn_pool_entry *entry = &pool->entries[i];
        if (entry->valid) {
            if (entry->qp) ibv_destroy_qp(entry->qp);
            if (entry->cq) ibv_destroy_cq(entry->cq);
            entry->valid = 0;
        }
    }

    pool->num_entries = 0;

    pthread_mutex_unlock(&pool->mutex);
    pthread_mutex_destroy(&pool->mutex);

    MESH_INFO("Connection pool destroyed (hits=%lu, misses=%lu)",
              pool->hits, pool->misses);
}

/*
 * Find an existing pooled connection to a peer
 * Returns NULL if not found
 */
struct mesh_conn_pool_entry* mesh_conn_pool_find(uint32_t peer_ip, struct mesh_nic *nic) {
    struct mesh_conn_pool *pool = &g_mesh_state.conn_pool;
    struct mesh_conn_pool_entry *found = NULL;

    pthread_mutex_lock(&pool->mutex);

    for (int i = 0; i < pool->num_entries; i++) {
        struct mesh_conn_pool_entry *entry = &pool->entries[i];
        if (entry->valid && entry->peer_ip == peer_ip && entry->nic == nic) {
            found = entry;
            break;
        }
    }

    pthread_mutex_unlock(&pool->mutex);
    return found;
}

/*
 * Acquire a pooled connection to a peer
 * Returns existing connection if available, NULL if none available
 */
struct mesh_conn_pool_entry* mesh_conn_pool_acquire(uint32_t peer_ip, struct mesh_nic *nic) {
    struct mesh_conn_pool *pool = &g_mesh_state.conn_pool;
    struct mesh_conn_pool_entry *entry = NULL;
    struct timespec ts;

    pthread_mutex_lock(&pool->mutex);

    // Look for existing connection to this peer
    for (int i = 0; i < pool->num_entries; i++) {
        struct mesh_conn_pool_entry *e = &pool->entries[i];
        if (e->valid && !e->in_use && e->peer_ip == peer_ip && e->nic == nic) {
            entry = e;
            break;
        }
    }

    if (entry) {
        entry->in_use = 1;
        entry->ref_count++;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        entry->last_used = ts.tv_sec;
        pool->hits++;

        char ip_str[INET_ADDRSTRLEN];
        mesh_uint_to_ip(peer_ip, ip_str, sizeof(ip_str));
        MESH_DEBUG("Pool hit: reusing connection to %s (QP %d, ref_count=%d)",
                   ip_str, entry->qp->qp_num, entry->ref_count);
    } else {
        pool->misses++;
    }

    pthread_mutex_unlock(&pool->mutex);
    return entry;
}

/*
 * Release a pooled connection back to the pool
 */
void mesh_conn_pool_release(struct mesh_conn_pool_entry *entry) {
    struct mesh_conn_pool *pool = &g_mesh_state.conn_pool;

    if (!entry) return;

    pthread_mutex_lock(&pool->mutex);

    entry->ref_count--;
    if (entry->ref_count <= 0) {
        entry->in_use = 0;
        entry->ref_count = 0;

        char ip_str[INET_ADDRSTRLEN];
        mesh_uint_to_ip(entry->peer_ip, ip_str, sizeof(ip_str));
        MESH_DEBUG("Pool release: connection to %s available for reuse", ip_str);
    }

    pthread_mutex_unlock(&pool->mutex);
}

/*
 * Add a new connection to the pool
 * Returns 0 on success, -1 if pool is full
 */
int mesh_conn_pool_add(uint32_t peer_ip, struct mesh_nic *nic,
                       struct ibv_qp *qp, struct ibv_cq *cq, uint32_t remote_qp_num) {
    struct mesh_conn_pool *pool = &g_mesh_state.conn_pool;
    struct mesh_conn_pool_entry *entry = NULL;
    struct timespec ts;
    int result = -1;

    pthread_mutex_lock(&pool->mutex);

    // Find a free slot or evict LRU entry
    if (pool->num_entries < MESH_CONN_POOL_SIZE) {
        entry = &pool->entries[pool->num_entries];
        pool->num_entries++;
    } else {
        // Find LRU entry that's not in use
        uint64_t oldest_time = UINT64_MAX;
        for (int i = 0; i < pool->num_entries; i++) {
            struct mesh_conn_pool_entry *e = &pool->entries[i];
            if (!e->in_use && e->last_used < oldest_time) {
                oldest_time = e->last_used;
                entry = e;
            }
        }

        if (entry) {
            // Evict old entry
            char ip_str[INET_ADDRSTRLEN];
            mesh_uint_to_ip(entry->peer_ip, ip_str, sizeof(ip_str));
            MESH_DEBUG("Pool evict: removing connection to %s", ip_str);

            if (entry->qp) ibv_destroy_qp(entry->qp);
            if (entry->cq) ibv_destroy_cq(entry->cq);
        }
    }

    if (entry) {
        clock_gettime(CLOCK_MONOTONIC, &ts);

        entry->valid = 1;
        entry->in_use = 1;
        entry->ref_count = 1;
        entry->peer_ip = peer_ip;
        entry->remote_qp_num = remote_qp_num;
        entry->nic = nic;
        entry->qp = qp;
        entry->cq = cq;
        entry->last_used = ts.tv_sec;

        char ip_str[INET_ADDRSTRLEN];
        mesh_uint_to_ip(peer_ip, ip_str, sizeof(ip_str));
        MESH_DEBUG("Pool add: new connection to %s (QP %d)", ip_str, qp->qp_num);

        result = 0;
    } else {
        MESH_WARN("Connection pool full, cannot add new entry");
    }

    pthread_mutex_unlock(&pool->mutex);
    return result;
}

/*
 * ============================================================================
 * Async Connect Implementation (TICKET-7)
 * ============================================================================
 *
 * Background thread for non-blocking connection establishment. This allows
 * mesh_connect() to return early while the handshake completes in the background,
 * improving performance when setting up connections to multiple peers.
 */

/*
 * Background thread for async connection establishment
 */
static void *async_connect_thread_func(void *arg) {
    (void)arg;
    struct mesh_async_connect_state *state = &g_mesh_state.async_connect;

    MESH_DEBUG("Async connect thread started");

    while (!state->thread_stop) {
        struct mesh_async_connect_req *req = NULL;

        pthread_mutex_lock(&state->mutex);

        // Wait for work
        while (state->queue_head == state->queue_tail && !state->thread_stop) {
            pthread_cond_wait(&state->cond, &state->mutex);
        }

        if (state->thread_stop) {
            pthread_mutex_unlock(&state->mutex);
            break;
        }

        // Get next request
        req = &state->queue[state->queue_head];
        if (req->valid && !req->complete) {
            pthread_mutex_unlock(&state->mutex);

            // Perform the connection (outside lock)
            char ip_str[INET_ADDRSTRLEN];
            mesh_uint_to_ip(req->peer_ip, ip_str, sizeof(ip_str));
            MESH_DEBUG("Async connect: processing request for %s", ip_str);

            // Create QP
            if (mesh_create_qp(req->nic, &req->qp, &req->cq) != 0) {
                req->error = 1;
                req->complete = 1;
                MESH_WARN("Async connect: failed to create QP for %s", ip_str);
                continue;
            }

            // Do handshake
            struct mesh_qp_info local_info, remote_info;
            memset(&local_info, 0, sizeof(local_info));
            local_info.qp_num = htonl(req->qp->qp_num);
            local_info.psn = htonl(0);
            local_info.ip = htonl(req->nic->ip_addr);
            local_info.gid_index = req->nic->gid_index;
            local_info.nic_idx = req->selected_addr->nic_idx;

            union ibv_gid our_gid;
            if (mesh_get_gid(req->nic, &our_gid) == 0) {
                memcpy(local_info.gid, our_gid.raw, 16);
            }

            uint32_t handshake_ip = ntohl(req->selected_addr->ip);
            if (mesh_send_handshake(handshake_ip, req->handle.handshake_port,
                                    &local_info, &remote_info) != 0) {
                req->error = 1;
                req->complete = 1;
                ibv_destroy_qp(req->qp);
                ibv_destroy_cq(req->cq);
                req->qp = NULL;
                req->cq = NULL;
                MESH_WARN("Async connect: handshake failed for %s", ip_str);
                continue;
            }

            usleep(1000);  // Small delay for acceptor's QP to be ready

            // Connect QP
            struct mesh_handle connect_handle;
            memset(&connect_handle, 0, sizeof(connect_handle));
            connect_handle.qp_num = ntohl(remote_info.qp_num);
            connect_handle.psn = ntohl(remote_info.psn);
            connect_handle.port_num = req->nic->port_num;
            connect_handle.mtu = req->nic->active_mtu;  // Use NIC's actual MTU (TICKET-9)

            // Construct peer GID
            union ibv_gid peer_gid;
            memset(&peer_gid, 0, sizeof(peer_gid));
            peer_gid.raw[10] = 0xff;
            peer_gid.raw[11] = 0xff;
            uint32_t remote_ip_for_gid = remote_info.ip ? remote_info.ip : req->selected_addr->ip;
            memcpy(&peer_gid.raw[12], &remote_ip_for_gid, 4);
            connect_handle.gid = peer_gid;

            if (mesh_connect_qp(req->qp, req->nic, &connect_handle) != 0) {
                req->error = 1;
                req->complete = 1;
                ibv_destroy_qp(req->qp);
                ibv_destroy_cq(req->cq);
                req->qp = NULL;
                req->cq = NULL;
                MESH_WARN("Async connect: QP connect failed for %s", ip_str);
                continue;
            }

            req->remote_qp_num = connect_handle.qp_num;
            req->complete = 1;
            req->error = 0;

            MESH_DEBUG("Async connect: completed for %s (QP %d -> %d)",
                       ip_str, req->qp->qp_num, req->remote_qp_num);

            pthread_mutex_lock(&state->mutex);
        }

        // Advance queue head
        state->queue_head = (state->queue_head + 1) % MESH_ASYNC_QUEUE_SIZE;
        pthread_mutex_unlock(&state->mutex);
    }

    MESH_DEBUG("Async connect thread stopped");
    return NULL;
}

/*
 * Initialize async connect state
 */
int mesh_async_connect_init(void) {
    struct mesh_async_connect_state *state = &g_mesh_state.async_connect;

    memset(state, 0, sizeof(*state));
    pthread_mutex_init(&state->mutex, NULL);
    pthread_cond_init(&state->cond, NULL);

    if (pthread_create(&state->thread, NULL, async_connect_thread_func, NULL) != 0) {
        MESH_WARN("Failed to create async connect thread");
        return -1;
    }

    state->thread_running = 1;
    MESH_INFO("Async connect thread initialized");
    return 0;
}

/*
 * Destroy async connect state
 */
void mesh_async_connect_destroy(void) {
    struct mesh_async_connect_state *state = &g_mesh_state.async_connect;

    if (state->thread_running) {
        pthread_mutex_lock(&state->mutex);
        state->thread_stop = 1;
        pthread_cond_broadcast(&state->cond);
        pthread_mutex_unlock(&state->mutex);

        pthread_join(state->thread, NULL);
        state->thread_running = 0;
    }

    // Clean up any pending requests
    for (int i = 0; i < MESH_ASYNC_QUEUE_SIZE; i++) {
        struct mesh_async_connect_req *req = &state->queue[i];
        if (req->valid && req->qp) {
            ibv_destroy_qp(req->qp);
        }
        if (req->valid && req->cq) {
            ibv_destroy_cq(req->cq);
        }
    }

    pthread_mutex_destroy(&state->mutex);
    pthread_cond_destroy(&state->cond);

    MESH_INFO("Async connect destroyed");
}

/*
 * Submit an async connect request
 * Returns request handle for polling, or NULL on error
 */
struct mesh_async_connect_req* mesh_async_connect_submit(struct mesh_handle *handle,
                                                          struct mesh_nic *nic,
                                                          struct mesh_addr_entry *addr,
                                                          void *send_comm) {
    struct mesh_async_connect_state *state = &g_mesh_state.async_connect;
    struct mesh_async_connect_req *req = NULL;

    pthread_mutex_lock(&state->mutex);

    // Find free slot
    int next_tail = (state->queue_tail + 1) % MESH_ASYNC_QUEUE_SIZE;
    if (next_tail != state->queue_head) {
        req = &state->queue[state->queue_tail];

        memset(req, 0, sizeof(*req));
        req->valid = 1;
        req->complete = 0;
        req->error = 0;
        memcpy(&req->handle, handle, sizeof(*handle));
        req->nic = nic;
        req->selected_addr = addr;
        req->peer_ip = ntohl(addr->ip);
        req->send_comm = send_comm;

        state->queue_tail = next_tail;
        pthread_cond_signal(&state->cond);

        char ip_str[INET_ADDRSTRLEN];
        mesh_uint_to_ip(req->peer_ip, ip_str, sizeof(ip_str));
        MESH_DEBUG("Async connect: submitted request for %s", ip_str);
    } else {
        MESH_WARN("Async connect queue full");
    }

    pthread_mutex_unlock(&state->mutex);
    return req;
}

/*
 * Poll an async connect request for completion
 * Returns 1 if complete (success or error), 0 if still pending
 */
int mesh_async_connect_poll(struct mesh_async_connect_req *req) {
    if (!req) return 1;
    return req->complete;
}

/*
 * ============================================================================
 * TCP Fallback Implementation (TICKET-4)
 * ============================================================================
 *
 * When RDMA/IB setup fails or is disabled via NCCL_MESH_DISABLE_RDMA=1,
 * we fall back to TCP sockets for data transfer. This allows the plugin
 * to work on systems without working RDMA hardware.
 */

/*
 * Initialize TCP fallback mode
 * Called when RDMA init fails or is disabled
 */
int mesh_tcp_init(void) {
    MESH_WARN("TCP fallback mode activated - RDMA not available or disabled");
    g_mesh_state.tcp_fallback_active = 1;

    // In TCP mode, we still need network interfaces for communication
    // Re-scan interfaces to populate NIC list with TCP-capable interfaces
    struct ifaddrs *ifaddr, *ifa;

    if (getifaddrs(&ifaddr) == -1) {
        MESH_WARN("Failed to get interface list for TCP fallback: %s", strerror(errno));
        return -1;
    }

    g_mesh_state.num_nics = 0;

    for (ifa = ifaddr; ifa != NULL && g_mesh_state.num_nics < MESH_MAX_NICS; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL) continue;
        if (ifa->ifa_addr->sa_family != AF_INET) continue;
        if (ifa->ifa_flags & IFF_LOOPBACK) continue;  // Skip loopback
        if (!(ifa->ifa_flags & IFF_UP)) continue;      // Skip down interfaces

        struct mesh_nic *nic = &g_mesh_state.nics[g_mesh_state.num_nics];
        memset(nic, 0, sizeof(*nic));

        struct sockaddr_in *addr = (struct sockaddr_in *)ifa->ifa_addr;
        struct sockaddr_in *netmask = (struct sockaddr_in *)ifa->ifa_netmask;

        nic->ip_addr = ntohl(addr->sin_addr.s_addr);
        nic->netmask = ntohl(netmask->sin_addr.s_addr);
        nic->subnet = nic->ip_addr & nic->netmask;

        strncpy(nic->if_name, ifa->ifa_name, sizeof(nic->if_name) - 1);
        strncpy(nic->dev_name, ifa->ifa_name, sizeof(nic->dev_name) - 1);  // Use if_name as dev_name

        char ip_str[INET_ADDRSTRLEN];
        mesh_uint_to_ip(nic->ip_addr, ip_str, sizeof(ip_str));
        MESH_INFO("TCP fallback: Found interface %s with IP %s", nic->if_name, ip_str);

        g_mesh_state.num_nics++;
    }

    freeifaddrs(ifaddr);

    if (g_mesh_state.num_nics == 0) {
        MESH_WARN("No network interfaces found for TCP fallback");
        return -1;
    }

    MESH_INFO("TCP fallback initialized with %d interfaces", g_mesh_state.num_nics);
    return 0;
}

/*
 * TCP listen - create a listening socket
 */
static ncclResult_t mesh_tcp_listen_impl(int dev, void *handle, void **listenComm) {
    (void)dev;

    struct mesh_handle *h = (struct mesh_handle *)handle;
    struct mesh_tcp_listen_comm *comm;
    int sock;
    struct sockaddr_in addr;
    socklen_t addrlen = sizeof(addr);
    int opt = 1;

    comm = calloc(1, sizeof(*comm));
    if (!comm) {
        return ncclSystemError;
    }

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        MESH_WARN("TCP listen: Failed to create socket: %s", strerror(errno));
        free(comm);
        return ncclSystemError;
    }

    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = 0;  // Let OS choose port

    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        MESH_WARN("TCP listen: Failed to bind: %s", strerror(errno));
        close(sock);
        free(comm);
        return ncclSystemError;
    }

    if (listen(sock, 16) < 0) {
        MESH_WARN("TCP listen: Failed to listen: %s", strerror(errno));
        close(sock);
        free(comm);
        return ncclSystemError;
    }

    if (getsockname(sock, (struct sockaddr *)&addr, &addrlen) < 0) {
        MESH_WARN("TCP listen: Failed to get socket name: %s", strerror(errno));
        close(sock);
        free(comm);
        return ncclSystemError;
    }

    comm->listen_sock = sock;
    comm->listen_port = ntohs(addr.sin_port);
    comm->listen_ip = INADDR_ANY;
    comm->ready = 1;

    // Fill handle for peer
    memset(h, 0, sizeof(*h));
    h->magic = MESH_HANDLE_MAGIC;
    h->handshake_port = comm->listen_port;
    h->num_addrs = 0;

    // Add all our addresses to the handle
    for (int i = 0; i < g_mesh_state.num_nics && h->num_addrs < MESH_MAX_ADDRS; i++) {
        struct mesh_nic *nic = &g_mesh_state.nics[i];
        struct mesh_addr_entry *entry = &h->addrs[h->num_addrs];

        entry->ip = htonl(nic->ip_addr);
        entry->mask = htonl(nic->netmask);
        entry->nic_idx = i;

        char ip_str[INET_ADDRSTRLEN];
        mesh_uint_to_ip(nic->ip_addr, ip_str, sizeof(ip_str));
        MESH_INFO("TCP listen: Advertising address %s on port %d", ip_str, comm->listen_port);

        h->num_addrs++;
    }

    MESH_INFO("TCP listen: Ready on port %d with %d addresses", comm->listen_port, h->num_addrs);

    *listenComm = comm;
    return ncclSuccess;
}

/*
 * TCP connect - connect to peer
 */
static ncclResult_t mesh_tcp_connect_impl(int dev, void *opaqueHandle, void **sendComm,
                                          ncclNetDeviceHandle_t **sendDevComm) {
    (void)dev;

    struct mesh_handle *handle = (struct mesh_handle *)opaqueHandle;
    struct mesh_tcp_send_comm *comm;
    struct mesh_nic *nic = NULL;
    struct mesh_addr_entry *selected_addr = NULL;

    if (handle->magic != MESH_HANDLE_MAGIC) {
        MESH_WARN("TCP connect: Invalid handle magic");
        return ncclInvalidArgument;
    }

    // Find a peer address we can reach
    for (int i = 0; i < handle->num_addrs; i++) {
        struct mesh_addr_entry *addr = &handle->addrs[i];
        uint32_t peer_ip = ntohl(addr->ip);

        nic = mesh_find_nic_for_ip(peer_ip);
        if (nic) {
            selected_addr = addr;
            break;
        }
    }

    if (!nic || !selected_addr) {
        MESH_WARN("TCP connect: No matching interface for peer");
        return ncclSystemError;
    }

    comm = calloc(1, sizeof(*comm));
    if (!comm) {
        return ncclSystemError;
    }

    // Create and connect socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        MESH_WARN("TCP connect: Failed to create socket: %s", strerror(errno));
        free(comm);
        return ncclSystemError;
    }

    // Set TCP_NODELAY for low latency
    int opt = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = selected_addr->ip;  // Already in network byte order
    addr.sin_port = htons(handle->handshake_port);

    // Retry connection with timeout
    int connected = 0;
    int retries = g_mesh_state.retry_count;
    int retry_delay_ms = 100;

    while (retries > 0 && !connected) {
        if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            connected = 1;
        } else if (errno == ECONNREFUSED || errno == ETIMEDOUT) {
            retries--;
            if (retries > 0) {
                usleep(retry_delay_ms * 1000);
                retry_delay_ms *= 2;
            }
        } else {
            break;
        }
    }

    if (!connected) {
        char ip_str[INET_ADDRSTRLEN];
        mesh_uint_to_ip(ntohl(selected_addr->ip), ip_str, sizeof(ip_str));
        MESH_WARN("TCP connect: Failed to connect to %s:%d: %s",
                  ip_str, handle->handshake_port, strerror(errno));
        close(sock);
        free(comm);
        return ncclSystemError;
    }

    comm->sock = sock;
    comm->remote_ip = ntohl(selected_addr->ip);
    comm->remote_port = handle->handshake_port;
    comm->connected = 1;

    char ip_str[INET_ADDRSTRLEN];
    mesh_uint_to_ip(comm->remote_ip, ip_str, sizeof(ip_str));
    MESH_INFO("TCP connect: Connected to %s:%d", ip_str, comm->remote_port);

    *sendComm = comm;
    if (sendDevComm) *sendDevComm = NULL;
    return ncclSuccess;
}

/*
 * TCP accept - accept incoming connection
 */
static ncclResult_t mesh_tcp_accept_impl(void *listenComm, void **recvComm,
                                         ncclNetDeviceHandle_t **recvDevComm) {
    struct mesh_tcp_listen_comm *lcomm = (struct mesh_tcp_listen_comm *)listenComm;
    struct mesh_tcp_recv_comm *comm;
    struct sockaddr_in addr;
    socklen_t addrlen = sizeof(addr);

    comm = calloc(1, sizeof(*comm));
    if (!comm) {
        return ncclSystemError;
    }

    // Set socket to non-blocking for timeout handling
    int flags = fcntl(lcomm->listen_sock, F_GETFL, 0);
    fcntl(lcomm->listen_sock, F_SETFL, flags | O_NONBLOCK);

    int conn_sock = -1;
    int timeout_ms = g_mesh_state.timeout_ms * 6;  // Same as RDMA accept timeout
    int elapsed = 0;

    while (elapsed < timeout_ms) {
        conn_sock = accept(lcomm->listen_sock, (struct sockaddr *)&addr, &addrlen);
        if (conn_sock >= 0) {
            break;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            usleep(10000);  // 10ms
            elapsed += 10;
        } else {
            break;
        }
    }

    // Restore blocking mode
    fcntl(lcomm->listen_sock, F_SETFL, flags);

    if (conn_sock < 0) {
        MESH_WARN("TCP accept: Failed to accept connection: %s", strerror(errno));
        free(comm);
        return ncclSystemError;
    }

    // Set TCP_NODELAY
    int opt = 1;
    setsockopt(conn_sock, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    comm->sock = conn_sock;
    comm->remote_ip = ntohl(addr.sin_addr.s_addr);
    comm->connected = 1;

    char ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &addr.sin_addr, ip_str, sizeof(ip_str));
    MESH_INFO("TCP accept: Accepted connection from %s", ip_str);

    *recvComm = comm;
    if (recvDevComm) *recvDevComm = NULL;
    return ncclSuccess;
}

/*
 * TCP memory registration - just record the buffer info (no actual MR)
 */
static ncclResult_t mesh_tcp_regMr(void *comm, void *data, size_t size, int type, void **mhandle) {
    (void)comm;
    (void)type;

    struct mesh_mr_handle *mrh = calloc(1, sizeof(*mrh));
    if (!mrh) {
        return ncclSystemError;
    }

    mrh->mr = NULL;  // No RDMA MR in TCP mode
    mrh->nic = NULL;
    mrh->addr = data;
    mrh->size = size;
    mrh->is_tcp = 1;

    *mhandle = mrh;
    return ncclSuccess;
}

/*
 * TCP send - send data over TCP socket with framing
 * TICKET-10: Made truly async to prevent deadlock on large transfers
 */
static ncclResult_t mesh_tcp_isend(void *sendComm, void *data, int size, int tag,
                                   void *mhandle, void **request) {
    struct mesh_tcp_send_comm *comm = (struct mesh_tcp_send_comm *)sendComm;
    struct mesh_tcp_request *req;
    (void)tag;
    (void)mhandle;

    if (!comm || comm->sock < 0) {
        MESH_WARN("TCP isend: Invalid comm");
        return ncclSystemError;
    }

    if (comm->peer_failed) {
        MESH_WARN("TCP isend: Peer already failed");
        return ncclSystemError;
    }

    req = calloc(1, sizeof(*req));
    if (!req) {
        return ncclSystemError;
    }
    __atomic_fetch_add(&g_mesh_state.tcp_requests_allocated, 1, __ATOMIC_RELAXED);

    req->used = 1;
    req->size = size;
    req->data = data;
    req->is_send = 1;
    req->comm = comm;
    req->done = 0;
    req->error = 0;
    req->offset = 0;
    req->header_sent = 0;

    // Set socket to non-blocking for async send
    int flags = fcntl(comm->sock, F_GETFL, 0);
    fcntl(comm->sock, F_SETFL, flags | O_NONBLOCK);

    // Try to send size header (4 bytes, network byte order)
    // TICKET-10: Handle EINTR and retry for transient errors
    uint32_t net_size = htonl(size);
    MESH_DEBUG("TCP isend: sending header size=%d (net_size=0x%08x)", size, net_size);
    ssize_t sent;
    do {
        sent = send(comm->sock, &net_size, sizeof(net_size), MSG_NOSIGNAL);
    } while (sent < 0 && errno == EINTR);

    if (sent < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // Can't even send header yet - return for polling
            fcntl(comm->sock, F_SETFL, flags);
            *request = req;
            return ncclSuccess;
        }
        MESH_WARN("TCP isend: Failed to send header: %s", strerror(errno));
        comm->peer_failed = 1;
        comm->last_errno = errno;
        fcntl(comm->sock, F_SETFL, flags);
        __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
        free(req);
        return ncclSystemError;
    }
    if (sent != sizeof(net_size)) {
        // Partial header send - treat as error (4 bytes should always fit)
        MESH_WARN("TCP isend: Partial header send (%zd/%zu)", sent, sizeof(net_size));
        comm->peer_failed = 1;
        fcntl(comm->sock, F_SETFL, flags);
        __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
        free(req);
        return ncclSystemError;
    }
    req->header_sent = 1;

    // Try to send as much data as possible without blocking
    // TICKET-10: Handle EINTR properly
    while (req->offset < (size_t)size) {
        do {
            sent = send(comm->sock, (char *)data + req->offset, size - req->offset, MSG_NOSIGNAL);
        } while (sent < 0 && errno == EINTR);

        if (sent <= 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Socket buffer full - return for async completion in test()
                fcntl(comm->sock, F_SETFL, flags);
                *request = req;
                return ncclSuccess;
            }
            MESH_WARN("TCP isend: Failed to send data: %s", strerror(errno));
            comm->peer_failed = 1;
            comm->last_errno = errno;
            req->error = errno;
            req->done = 1;
            fcntl(comm->sock, F_SETFL, flags);
            *request = req;
            return ncclSystemError;
        }
        req->offset += sent;
    }

    // All data sent
    fcntl(comm->sock, F_SETFL, flags);
    req->done = 1;
    *request = req;
    return ncclSuccess;
}

/*
 * TCP receive - receive data over TCP socket with framing
 * TICKET-10: Made truly async like isend to prevent deadlock
 */
static ncclResult_t mesh_tcp_irecv(void *recvComm, int n, void **data, int *sizes,
                                   int *tags, void **mhandles, void **request) {
    struct mesh_tcp_recv_comm *comm = (struct mesh_tcp_recv_comm *)recvComm;
    struct mesh_tcp_request *req;
    (void)tags;
    (void)mhandles;

    if (n != 1) {
        MESH_WARN("TCP irecv: Only n=1 supported");
        return ncclInternalError;
    }

    if (!comm || comm->sock < 0) {
        MESH_WARN("TCP irecv: Invalid comm");
        return ncclSystemError;
    }

    if (comm->peer_failed) {
        MESH_WARN("TCP irecv: Peer already failed");
        return ncclSystemError;
    }

    req = calloc(1, sizeof(*req));
    if (!req) {
        return ncclSystemError;
    }
    __atomic_fetch_add(&g_mesh_state.tcp_requests_allocated, 1, __ATOMIC_RELAXED);

    req->used = 1;
    req->size = sizes[0];
    req->data = data[0];
    req->is_send = 0;
    req->comm = comm;
    req->done = 0;
    req->error = 0;
    req->offset = 0;
    req->header_recvd = 0;
    req->msg_size = 0;

    // Set socket to non-blocking for async receive
    int flags = fcntl(comm->sock, F_GETFL, 0);
    fcntl(comm->sock, F_SETFL, flags | O_NONBLOCK);

    // Try to peek at the size header first to check availability
    // TICKET-10: Handle EINTR properly
    uint32_t net_size;
    ssize_t recvd;
    do {
        recvd = recv(comm->sock, &net_size, sizeof(net_size), MSG_PEEK);
    } while (recvd < 0 && errno == EINTR);

    if (recvd == 0) {
        // Connection closed
        MESH_WARN("TCP irecv: Connection closed by peer");
        comm->peer_failed = 1;
        fcntl(comm->sock, F_SETFL, flags);
        __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
        free(req);
        return ncclSystemError;
    } else if (recvd < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        // No data yet - will poll in test
        fcntl(comm->sock, F_SETFL, flags);
        *request = req;
        return ncclSuccess;
    } else if (recvd < (ssize_t)sizeof(net_size)) {
        // Partial header or error - wait for more data in test()
        if (recvd > 0) {
            // Partial header available, poll again later
            fcntl(comm->sock, F_SETFL, flags);
            *request = req;
            return ncclSuccess;
        }
        MESH_WARN("TCP irecv: Failed to receive header: %s", strerror(errno));
        comm->peer_failed = 1;
        comm->last_errno = errno;
        fcntl(comm->sock, F_SETFL, flags);
        __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
        free(req);
        return ncclSystemError;
    }

    // Full header available via peek, now consume it
    do {
        recvd = recv(comm->sock, &net_size, sizeof(net_size), 0);
    } while (recvd < 0 && errno == EINTR);

    MESH_DEBUG("TCP irecv: received header net_size=0x%08x (raw bytes: %02x %02x %02x %02x)",
               net_size,
               ((unsigned char*)&net_size)[0], ((unsigned char*)&net_size)[1],
               ((unsigned char*)&net_size)[2], ((unsigned char*)&net_size)[3]);

    if (recvd != sizeof(net_size)) {
        MESH_WARN("TCP irecv: Failed to consume header: %s", strerror(errno));
        comm->peer_failed = 1;
        comm->last_errno = errno;
        fcntl(comm->sock, F_SETFL, flags);
        __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
        free(req);
        return ncclSystemError;
    }

    uint32_t msg_size = ntohl(net_size);
    MESH_DEBUG("TCP irecv: msg_size=%u (after ntohl), buffer_size=%d", msg_size, sizes[0]);
    if (msg_size > (uint32_t)sizes[0]) {
        MESH_WARN("TCP irecv: Message too large (%u > %d)", msg_size, sizes[0]);
        fcntl(comm->sock, F_SETFL, flags);
        __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
        free(req);
        return ncclSystemError;
    }

    req->header_recvd = 1;
    req->msg_size = msg_size;

    // Try to receive as much data as possible without blocking
    // TICKET-10: Handle EINTR properly
    while (req->offset < msg_size) {
        do {
            recvd = recv(comm->sock, (char *)data[0] + req->offset, msg_size - req->offset, 0);
        } while (recvd < 0 && errno == EINTR);

        if (recvd == 0) {
            // Connection closed before all data received
            MESH_WARN("TCP irecv: Connection closed during data receive");
            comm->peer_failed = 1;
            req->error = ECONNRESET;
            req->done = 1;
            fcntl(comm->sock, F_SETFL, flags);
            *request = req;
            return ncclSystemError;
        } else if (recvd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // No more data available - return for async completion in test()
                fcntl(comm->sock, F_SETFL, flags);
                *request = req;
                return ncclSuccess;
            }
            MESH_WARN("TCP irecv: Failed to receive data: %s", strerror(errno));
            comm->peer_failed = 1;
            comm->last_errno = errno;
            req->error = errno;
            req->done = 1;
            fcntl(comm->sock, F_SETFL, flags);
            *request = req;
            return ncclSystemError;
        }
        req->offset += recvd;
    }

    // All data received
    fcntl(comm->sock, F_SETFL, flags);
    req->size = msg_size;
    req->done = 1;
    *request = req;
    return ncclSuccess;
}

/*
 * TCP test - check if request is complete
 */
static ncclResult_t mesh_tcp_test_impl(void *request, int *done, int *sizes) {
    struct mesh_tcp_request *req = (struct mesh_tcp_request *)request;

    if (!req) {
        *done = 1;
        return ncclSuccess;
    }

    if (req->done) {
        *done = 1;
        if (sizes) *sizes = req->size;
        ncclResult_t result = req->error ? ncclSystemError : ncclSuccess;
        __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED);
        free(req);  // TICKET-8: Free completed TCP request to prevent memory leak
        return result;
    }

    // TICKET-10: For incomplete receive, continue receiving data asynchronously
    if (!req->is_send && req->comm) {
        struct mesh_tcp_recv_comm *comm = (struct mesh_tcp_recv_comm *)req->comm;
        ssize_t recvd;

        // Set socket to non-blocking
        int flags = fcntl(comm->sock, F_GETFL, 0);
        fcntl(comm->sock, F_SETFL, flags | O_NONBLOCK);

        // If header not received yet, try to receive it
        if (!req->header_recvd) {
            uint32_t net_size;
            do {
                recvd = recv(comm->sock, &net_size, sizeof(net_size), MSG_PEEK);
            } while (recvd < 0 && errno == EINTR);

            if (recvd == (ssize_t)sizeof(net_size)) {
                // Header available via peek - consume it
                do {
                    recvd = recv(comm->sock, &net_size, sizeof(net_size), 0);
                } while (recvd < 0 && errno == EINTR);

                if (recvd != sizeof(net_size)) {
                    MESH_WARN("TCP test: Failed to consume header: %s", strerror(errno));
                    req->error = errno ? errno : EPROTO;
                    req->done = 1;
                    *done = 1;
                    fcntl(comm->sock, F_SETFL, flags);
                    __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
                    __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED);
                    free(req);
                    return ncclSystemError;
                }

                uint32_t msg_size = ntohl(net_size);
                if (msg_size > req->size) {
                    MESH_WARN("TCP test: Message too large (%u > %zu)", msg_size, req->size);
                    req->error = EMSGSIZE;
                    req->done = 1;
                    *done = 1;
                    fcntl(comm->sock, F_SETFL, flags);
                    __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
                    __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED);
                    free(req);
                    return ncclSystemError;
                }

                req->header_recvd = 1;
                req->msg_size = msg_size;
                req->offset = 0;
            } else if (recvd < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                // Header not ready yet
                fcntl(comm->sock, F_SETFL, flags);
                *done = 0;
                return ncclSuccess;
            } else if (recvd == 0) {
                // Connection closed
                MESH_WARN("TCP test: Connection closed by peer");
                req->error = ECONNRESET;
                req->done = 1;
                *done = 1;
                fcntl(comm->sock, F_SETFL, flags);
                __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
                __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED);
                free(req);
                return ncclSystemError;
            } else if (recvd > 0) {
                // Partial header - wait for more
                fcntl(comm->sock, F_SETFL, flags);
                *done = 0;
                return ncclSuccess;
            } else {
                // Error
                MESH_WARN("TCP test: Failed to receive header: %s", strerror(errno));
                req->error = errno;
                req->done = 1;
                *done = 1;
                fcntl(comm->sock, F_SETFL, flags);
                __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
                __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED);
                free(req);
                return ncclSystemError;
            }
        }

        // Continue receiving data from where we left off
        while (req->offset < req->msg_size) {
            do {
                recvd = recv(comm->sock, (char *)req->data + req->offset,
                            req->msg_size - req->offset, 0);
            } while (recvd < 0 && errno == EINTR);

            if (recvd == 0) {
                // Connection closed before all data received
                MESH_WARN("TCP test: Connection closed during data receive");
                req->error = ECONNRESET;
                req->done = 1;
                *done = 1;
                fcntl(comm->sock, F_SETFL, flags);
                __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
                __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED);
                free(req);
                return ncclSystemError;
            } else if (recvd < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // No more data available - return for more polling
                    fcntl(comm->sock, F_SETFL, flags);
                    *done = 0;
                    return ncclSuccess;
                }
                MESH_WARN("TCP test: Failed to receive data: %s", strerror(errno));
                req->error = errno;
                req->done = 1;
                *done = 1;
                fcntl(comm->sock, F_SETFL, flags);
                __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
                __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED);
                free(req);
                return ncclSystemError;
            }
            req->offset += recvd;
        }

        // All data received
        fcntl(comm->sock, F_SETFL, flags);
        req->size = req->msg_size;
        req->done = 1;
        *done = 1;
        if (sizes) *sizes = req->msg_size;
        __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED);
        free(req);
        return ncclSuccess;
    }

    // TICKET-10: For incomplete send, continue sending data
    if (req->is_send && req->comm) {
        struct mesh_tcp_send_comm *comm = (struct mesh_tcp_send_comm *)req->comm;
        ssize_t sent;

        // Set socket to non-blocking
        int flags = fcntl(comm->sock, F_GETFL, 0);
        fcntl(comm->sock, F_SETFL, flags | O_NONBLOCK);

        // If header not sent yet, try to send it
        if (!req->header_sent) {
            uint32_t net_size = htonl(req->size);
            do {
                sent = send(comm->sock, &net_size, sizeof(net_size), MSG_NOSIGNAL);
            } while (sent < 0 && errno == EINTR);

            if (sent < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    fcntl(comm->sock, F_SETFL, flags);
                    *done = 0;
                    return ncclSuccess;
                }
                MESH_WARN("TCP test: Failed to send header: %s", strerror(errno));
                comm->peer_failed = 1;
                req->error = errno;
                req->done = 1;
                *done = 1;
                fcntl(comm->sock, F_SETFL, flags);
                __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
                __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED);
                free(req);
                return ncclSystemError;
            }
            if (sent != sizeof(net_size)) {
                MESH_WARN("TCP test: Partial header send");
                comm->peer_failed = 1;
                req->error = EPROTO;
                req->done = 1;
                *done = 1;
                fcntl(comm->sock, F_SETFL, flags);
                __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
                __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED);
                free(req);
                return ncclSystemError;
            }
            req->header_sent = 1;
        }

        // Continue sending data from where we left off
        while (req->offset < req->size) {
            do {
                sent = send(comm->sock, (char *)req->data + req->offset,
                           req->size - req->offset, MSG_NOSIGNAL);
            } while (sent < 0 && errno == EINTR);

            if (sent <= 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // Socket buffer full - return for more polling
                    fcntl(comm->sock, F_SETFL, flags);
                    *done = 0;
                    return ncclSuccess;
                }
                MESH_WARN("TCP test: Failed to send data: %s", strerror(errno));
                comm->peer_failed = 1;
                req->error = errno;
                req->done = 1;
                *done = 1;
                fcntl(comm->sock, F_SETFL, flags);
                __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
                __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED);
                free(req);
                return ncclSystemError;
            }
            req->offset += sent;
        }

        // All data sent
        fcntl(comm->sock, F_SETFL, flags);
        req->done = 1;
        *done = 1;
        if (sizes) *sizes = req->size;
        __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED);
        free(req);
        return ncclSuccess;
    }

    // Check if request was completed in isend/irecv (synchronous completion)
    if (req->done) {
        *done = 1;
        if (sizes) *sizes = req->size;
        int had_error = req->error;  // Save before free
        __atomic_fetch_add(&g_mesh_state.tcp_requests_freed, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED);
        free(req);  // TICKET-8: Free synchronously completed request
        return had_error ? ncclSystemError : ncclSuccess;
    }

    *done = 0;
    return ncclSuccess;
}

/*
 * TCP close send
 */
static ncclResult_t mesh_tcp_closeSend(void *sendComm) {
    struct mesh_tcp_send_comm *comm = (struct mesh_tcp_send_comm *)sendComm;

    if (comm) {
        if (comm->sock >= 0) {
            close(comm->sock);
        }
        free(comm);
    }

    return ncclSuccess;
}

/*
 * TCP close receive
 */
static ncclResult_t mesh_tcp_closeRecv(void *recvComm) {
    struct mesh_tcp_recv_comm *comm = (struct mesh_tcp_recv_comm *)recvComm;

    if (comm) {
        if (comm->sock >= 0) {
            close(comm->sock);
        }
        free(comm);
    }

    return ncclSuccess;
}

/*
 * TCP close listen
 */
static ncclResult_t mesh_tcp_closeListen(void *listenComm) {
    struct mesh_tcp_listen_comm *comm = (struct mesh_tcp_listen_comm *)listenComm;

    if (comm) {
        if (comm->listen_sock >= 0) {
            close(comm->listen_sock);
        }
        free(comm);
    }

    return ncclSuccess;
}

/*
 * ============================================================================
 * NCCL Plugin API Implementation
 * ============================================================================
 */

static ncclResult_t mesh_init(ncclDebugLogger_t logFunction) {
    if (g_mesh_state.initialized) {
        return ncclSuccess;
    }

    g_mesh_state.log_fn = logFunction;

    // Read configuration from environment
    const char *env_val;

    // NCCL_MESH_GID_INDEX: RoCE GID index (default: 3)
    env_val = getenv("NCCL_MESH_GID_INDEX");
    g_mesh_state.gid_index = env_val ? atoi(env_val) : 3;

    // NCCL_MESH_DEBUG: Debug level 0=off, 1=info, 2=verbose (default: 0)
    env_val = getenv("NCCL_MESH_DEBUG");
    g_mesh_state.debug_level = env_val ? atoi(env_val) : 0;

    // NCCL_MESH_FAST_FAIL: Enable fast failure detection (default: 0)
    env_val = getenv("NCCL_MESH_FAST_FAIL");
    g_mesh_state.fast_fail = env_val ? atoi(env_val) : 0;

    // NCCL_MESH_TIMEOUT_MS: Connection timeout in milliseconds (default: 5000)
    env_val = getenv("NCCL_MESH_TIMEOUT_MS");
    g_mesh_state.timeout_ms = env_val ? atoi(env_val) : 5000;
    if (g_mesh_state.timeout_ms < 100) g_mesh_state.timeout_ms = 100;  // Minimum 100ms

    // NCCL_MESH_RETRY_COUNT: Number of retry attempts (default: 3)
    env_val = getenv("NCCL_MESH_RETRY_COUNT");
    g_mesh_state.retry_count = env_val ? atoi(env_val) : 3;
    if (g_mesh_state.retry_count < 1) g_mesh_state.retry_count = 1;  // Minimum 1

    // NCCL_MESH_DISABLE_RDMA: Force TCP fallback (default: 0)
    env_val = getenv("NCCL_MESH_DISABLE_RDMA");
    g_mesh_state.disable_rdma = env_val ? atoi(env_val) : 0;

    // NCCL_MESH_CONN_POOL: Enable connection pooling (default: 1) (TICKET-6)
    env_val = getenv("NCCL_MESH_CONN_POOL");
    g_mesh_state.enable_conn_pool = env_val ? atoi(env_val) : 1;

    // NCCL_MESH_ASYNC_CONNECT: Enable async connection (default: 1) (TICKET-7)
    env_val = getenv("NCCL_MESH_ASYNC_CONNECT");
    g_mesh_state.enable_async_connect = env_val ? atoi(env_val) : 1;

    // Log configuration (always shown at init, regardless of debug level)
    MESH_LOG(NCCL_LOG_INFO, "MESH Initializing: gid=%d debug=%d fast_fail=%d timeout=%dms retries=%d "
             "disable_rdma=%d conn_pool=%d async_connect=%d",
             g_mesh_state.gid_index, g_mesh_state.debug_level, g_mesh_state.fast_fail,
             g_mesh_state.timeout_ms, g_mesh_state.retry_count, g_mesh_state.disable_rdma,
             g_mesh_state.enable_conn_pool, g_mesh_state.enable_async_connect);

    // Verify handle struct size fits in NCCL limits (NCCL_NET_HANDLE_MAXSIZE = 128)
    MESH_LOG(NCCL_LOG_INFO, "MESH Handle size: %zu bytes (max 128)", sizeof(struct mesh_handle));
    if (sizeof(struct mesh_handle) > 128) {
        MESH_WARN("CRITICAL: mesh_handle size %zu > 128 bytes! Handle will be truncated!",
                  sizeof(struct mesh_handle));
        return ncclInternalError;
    }

    // Check if TCP fallback is forced (TICKET-4)
    if (g_mesh_state.disable_rdma) {
        MESH_WARN("NCCL_MESH_DISABLE_RDMA=1: Forcing TCP fallback mode");
        if (mesh_tcp_init() != 0) {
            MESH_WARN("TCP fallback initialization failed");
            return ncclSystemError;
        }
        g_mesh_state.initialized = 1;
        MESH_INFO("Mesh plugin initialized in TCP fallback mode with %d interfaces", g_mesh_state.num_nics);
        return ncclSuccess;
    }

    // Try RDMA initialization
    if (mesh_init_nics() != 0) {
        MESH_WARN("Failed to initialize RDMA NICs, attempting TCP fallback");
        g_mesh_state.rdma_init_failed = 1;

        // Attempt TCP fallback (TICKET-4)
        if (mesh_tcp_init() != 0) {
            MESH_WARN("Both RDMA and TCP fallback initialization failed");
            return ncclSystemError;
        }
        g_mesh_state.initialized = 1;
        MESH_INFO("Mesh plugin initialized in TCP fallback mode with %d interfaces", g_mesh_state.num_nics);
        return ncclSuccess;
    }

    // Initialize connection pool if enabled (TICKET-6)
    if (g_mesh_state.enable_conn_pool) {
        mesh_conn_pool_init();
    }

    // Initialize async connect if enabled (TICKET-7)
    if (g_mesh_state.enable_async_connect) {
        mesh_async_connect_init();
    }

    g_mesh_state.initialized = 1;
    MESH_INFO("Mesh plugin initialized with %d NICs (RDMA mode)", g_mesh_state.num_nics);

    return ncclSuccess;
}

static ncclResult_t mesh_devices(int *ndev) {
    *ndev = g_mesh_state.num_nics;
    return ncclSuccess;
}

static ncclResult_t mesh_getProperties(int dev, ncclNetProperties_v8_t *props) {
    if (dev < 0 || dev >= g_mesh_state.num_nics) {
        return ncclInvalidArgument;
    }
    
    struct mesh_nic *nic = &g_mesh_state.nics[dev];
    
    memset(props, 0, sizeof(*props));
    props->name = nic->dev_name;
    props->pciPath = nic->pci_path;
    props->guid = 0;  // TODO: Get actual GUID
    props->ptrSupport = NCCL_PTR_HOST;  // Only host memory for now (no GPUDirect RDMA)
    props->speed = 100000;  // 100 Gbps
    props->port = nic->port_num;
    props->latency = 1.0;
    props->maxComms = nic->max_qp;
    props->maxRecvs = 1;
    props->netDeviceType = NCCL_NET_DEVICE_HOST;
    props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
    props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
    
    return ncclSuccess;
}

static ncclResult_t mesh_listen(int dev, void *handle, void **listenComm) {
    // Dispatch to TCP if in fallback mode (TICKET-4)
    if (g_mesh_state.tcp_fallback_active) {
        return mesh_tcp_listen_impl(dev, handle, listenComm);
    }

    (void)dev;  // We listen on ALL NICs, not just the requested one

    struct mesh_handle *h = (struct mesh_handle *)handle;
    struct mesh_listen_comm *comm;
    union ibv_gid gid;
    
    // Allocate listen comm
    comm = calloc(1, sizeof(*comm));
    if (!comm) {
        return ncclSystemError;
    }
    
    comm->num_qps = 0;
    comm->psn = 0;
    comm->handshake_sock = -1;
    
    // Create QP on EACH NIC
    for (int i = 0; i < g_mesh_state.num_nics && i < MESH_MAX_NICS; i++) {
        struct mesh_nic *nic = &g_mesh_state.nics[i];
        struct ibv_qp *qp = NULL;
        struct ibv_cq *cq = NULL;
        
        if (mesh_create_qp(nic, &qp, &cq) != 0) {
            MESH_WARN("Failed to create QP on NIC %s, skipping", nic->dev_name);
            continue;
        }
        
        comm->qps[comm->num_qps].nic = nic;
        comm->qps[comm->num_qps].qp = qp;
        comm->qps[comm->num_qps].cq = cq;
        comm->num_qps++;
        
        char ip_str[INET_ADDRSTRLEN];
        mesh_uint_to_ip(nic->ip_addr, ip_str, sizeof(ip_str));
        MESH_INFO("listen: Created QP %d on %s (IP=%s)", qp->qp_num, nic->dev_name, ip_str);
    }
    
    if (comm->num_qps == 0) {
        MESH_WARN("Failed to create any QPs");
        free(comm);
        return ncclSystemError;
    }
    
    // Create handshake socket - bind to all interfaces so any peer can reach us
    comm->handshake_ip = INADDR_ANY;
    comm->handshake_sock = mesh_create_handshake_socket(INADDR_ANY, &comm->handshake_port);
    if (comm->handshake_sock < 0) {
        MESH_WARN("Failed to create handshake socket");
        // Continue without handshake - will fail at accept
    }
    
    // Initialize handshake queue and thread
    pthread_mutex_init(&comm->queue_mutex, NULL);
    pthread_cond_init(&comm->queue_cond, NULL);
    comm->queue_head = 0;
    comm->queue_tail = 0;
    comm->thread_stop = 0;
    comm->thread_running = 0;
    
    // Start handshake thread
    if (comm->handshake_sock >= 0) {
        if (pthread_create(&comm->handshake_thread, NULL, handshake_thread_func, comm) == 0) {
            comm->thread_running = 1;
        } else {
            MESH_WARN("Failed to start handshake thread");
        }
    }
    
    // Fill handle with ALL our addresses
    memset(h, 0, sizeof(*h));
    h->magic = MESH_HANDLE_MAGIC;
    h->num_addrs = 0;
    h->psn = comm->psn;
    h->port_num = 1;
    h->mtu = comm->qps[0].nic->active_mtu;  // Use NIC's actual MTU (TICKET-9)
    h->handshake_port = comm->handshake_port;
    // Store first NIC IP in handle - but connector will use selected_addr->ip for handshake
    h->handshake_ip = htonl(comm->qps[0].nic->ip_addr);
    
    // Get GID from first NIC for the primary GID field (TICKET-5: use cached GID)
    struct mesh_nic *primary_nic = comm->qps[0].nic;
    if (mesh_get_gid(primary_nic, &gid) == 0) {
        h->gid = gid;
    }
    
    // Add all NIC addresses to the handle
    for (int i = 0; i < comm->num_qps && h->num_addrs < MESH_MAX_ADDRS; i++) {
        struct mesh_nic *nic = comm->qps[i].nic;
        struct mesh_addr_entry *entry = &h->addrs[h->num_addrs];
        
        entry->ip = htonl(nic->ip_addr);
        entry->mask = htonl(nic->netmask);
        entry->qp_num = comm->qps[i].qp->qp_num;
        entry->nic_idx = i;
        entry->gid_index = nic->gid_index;
        
        char ip_str[INET_ADDRSTRLEN];
        mesh_uint_to_ip(nic->ip_addr, ip_str, sizeof(ip_str));
        MESH_INFO("listen: Advertising address %d: %s (QP %d)", 
                  h->num_addrs, ip_str, entry->qp_num);
        
        h->num_addrs++;
    }
    
    MESH_INFO("listen: Ready with %d addresses on %d QPs, handshake port %d", 
              h->num_addrs, comm->num_qps, comm->handshake_port);
    
    *listenComm = comm;
    return ncclSuccess;
}

/*
 * connect() - THE KEY FUNCTION
 *
 * Search through peer's advertised addresses to find one on a subnet we can reach
 */
static ncclResult_t mesh_connect(int dev, void *opaqueHandle, void **sendComm,
                                 ncclNetDeviceHandle_t **sendDevComm) {
    // Dispatch to TCP if in fallback mode (TICKET-4)
    if (g_mesh_state.tcp_fallback_active) {
        return mesh_tcp_connect_impl(dev, opaqueHandle, sendComm, sendDevComm);
    }

    (void)dev;  // We pick the right NIC based on subnet match

    struct mesh_handle *handle = (struct mesh_handle *)opaqueHandle;
    struct mesh_send_comm *comm;
    struct mesh_nic *nic = NULL;
    struct mesh_addr_entry *selected_addr = NULL;
    
    // Validate handle
    if (handle->magic != MESH_HANDLE_MAGIC) {
        MESH_WARN("Invalid handle magic: 0x%x (expected 0x%x)", handle->magic, MESH_HANDLE_MAGIC);
        MESH_WARN("Handle size: %zu bytes, raw bytes:", sizeof(struct mesh_handle));
        // Hex dump first 64 bytes for debugging
        uint8_t *raw = (uint8_t *)handle;
        for (int i = 0; i < 64 && i < (int)sizeof(struct mesh_handle); i += 16) {
            MESH_WARN("  %02d: %02x %02x %02x %02x %02x %02x %02x %02x  %02x %02x %02x %02x %02x %02x %02x %02x",
                      i, raw[i], raw[i+1], raw[i+2], raw[i+3], raw[i+4], raw[i+5], raw[i+6], raw[i+7],
                      raw[i+8], raw[i+9], raw[i+10], raw[i+11], raw[i+12], raw[i+13], raw[i+14], raw[i+15]);
        }
        return ncclInvalidArgument;
    }
    
    MESH_INFO("connect: Peer advertised %d addresses", handle->num_addrs);
    
    // Search through peer's addresses to find one we can reach
    for (int i = 0; i < handle->num_addrs; i++) {
        struct mesh_addr_entry *addr = &handle->addrs[i];
        uint32_t peer_ip = ntohl(addr->ip);
        
        char ip_str[INET_ADDRSTRLEN];
        mesh_uint_to_ip(peer_ip, ip_str, sizeof(ip_str));
        MESH_DEBUG("connect: Checking peer address %d: %s", i, ip_str);
        
        // Find local NIC on same subnet
        nic = mesh_find_nic_for_ip(peer_ip);
        if (nic) {
            selected_addr = addr;
            MESH_INFO("connect: Found matching NIC %s for peer %s", nic->dev_name, ip_str);
            break;
        }
    }
    
    if (!nic || !selected_addr) {
        MESH_WARN("connect: No local NIC found on same subnet as any peer address");
        for (int i = 0; i < handle->num_addrs; i++) {
            char ip_str[INET_ADDRSTRLEN];
            mesh_uint_to_ip(ntohl(handle->addrs[i].ip), ip_str, sizeof(ip_str));
            MESH_WARN("  Peer address %d: %s", i, ip_str);
        }
        return ncclSystemError;
    }
    
    uint32_t peer_ip = ntohl(selected_addr->ip);
    char peer_ip_str[INET_ADDRSTRLEN];
    mesh_uint_to_ip(peer_ip, peer_ip_str, sizeof(peer_ip_str));

    // Allocate send comm
    comm = calloc(1, sizeof(*comm));
    if (!comm) {
        return ncclSystemError;
    }

    comm->nic = nic;
    comm->peer_ip = peer_ip;

    // TICKET-6: Check connection pool for existing connection to this peer
    if (g_mesh_state.enable_conn_pool) {
        struct mesh_conn_pool_entry *pool_entry = mesh_conn_pool_acquire(peer_ip, nic);
        if (pool_entry) {
            // Pool hit - reuse existing connection
            comm->qp = pool_entry->qp;
            comm->cq = pool_entry->cq;
            comm->pool_entry = pool_entry;
            comm->remote_qp_num = pool_entry->remote_qp_num;
            comm->connected = 1;

            MESH_INFO("connect: Pool hit - reusing connection to %s (QP %d -> %d)",
                      peer_ip_str, comm->qp->qp_num, comm->remote_qp_num);

            *sendComm = comm;
            if (sendDevComm) *sendDevComm = NULL;
            return ncclSuccess;
        }
        MESH_DEBUG("connect: Pool miss for %s, creating new connection", peer_ip_str);
    }

    // Create QP on the selected NIC
    if (mesh_create_qp(nic, &comm->qp, &comm->cq) != 0) {
        free(comm);
        return ncclSystemError;
    }


    // Do handshake FIRST to get accept's QP number
    struct mesh_qp_info remote_qp_info;
    memset(&remote_qp_info, 0, sizeof(remote_qp_info));

    if (handle->handshake_port > 0) {

        struct mesh_qp_info local_info;
        memset(&local_info, 0, sizeof(local_info));
        local_info.qp_num = htonl(comm->qp->qp_num);
        local_info.psn = htonl(0);  // Our PSN
        local_info.ip = htonl(nic->ip_addr);
        local_info.gid_index = nic->gid_index;
        local_info.nic_idx = selected_addr->nic_idx;  // Which of listener's NICs we want

        // Copy our GID (TICKET-5: use cached GID)
        union ibv_gid our_gid;
        if (mesh_get_gid(nic, &our_gid) == 0) {
            memcpy(local_info.gid, our_gid.raw, 16);
        }

        // Bidirectional handshake - send our info, receive accept's info
        uint32_t handshake_ip = ntohl(selected_addr->ip);

        char hs_ip_str[INET_ADDRSTRLEN];
        mesh_uint_to_ip(handshake_ip, hs_ip_str, sizeof(hs_ip_str));

        if (mesh_send_handshake(handshake_ip, handle->handshake_port, &local_info, &remote_qp_info) != 0) {
            MESH_WARN("connect: Bidirectional handshake failed");
            ibv_destroy_qp(comm->qp);
            ibv_destroy_cq(comm->cq);
            free(comm);
            return ncclSystemError;
        }

        // Small delay to let acceptor's QP transition complete
        // The acceptor connects its QP before sending the response, but there's a small
        // window where the QP state may not be fully visible to us yet
        usleep(1000);  // 1ms

    } else {
        MESH_WARN("connect: No handshake port - using listen QP (will likely fail)");
        remote_qp_info.qp_num = htonl(selected_addr->qp_num);
        remote_qp_info.psn = htonl(handle->psn);
        remote_qp_info.ip = selected_addr->ip;
    }

    // Now connect our QP to the ACCEPT's QP (from handshake response)
    struct mesh_handle connect_handle;
    memset(&connect_handle, 0, sizeof(connect_handle));
    connect_handle.qp_num = ntohl(remote_qp_info.qp_num);  // Accept's new QP!
    connect_handle.psn = ntohl(remote_qp_info.psn);
    connect_handle.lid = 0;  // RoCE uses GID, not LID
    connect_handle.port_num = nic->port_num;
    // Negotiate MTU: use minimum of local and remote (TICKET-9)
    connect_handle.mtu = (handle->mtu && handle->mtu < nic->active_mtu)
                         ? handle->mtu : nic->active_mtu;

    // Construct peer GID from their IP
    union ibv_gid peer_gid;
    memset(&peer_gid, 0, sizeof(peer_gid));
    peer_gid.raw[10] = 0xff;
    peer_gid.raw[11] = 0xff;
    uint32_t remote_ip_for_gid = remote_qp_info.ip;  // Already in network byte order from handshake
    if (remote_ip_for_gid == 0) {
        remote_ip_for_gid = selected_addr->ip;  // Fallback
    }
    memcpy(&peer_gid.raw[12], &remote_ip_for_gid, 4);
    connect_handle.gid = peer_gid;


    // Connect QP to remote
    if (mesh_connect_qp(comm->qp, nic, &connect_handle) != 0) {
        MESH_WARN("connect: Failed to connect QP to peer");
        ibv_destroy_qp(comm->qp);
        ibv_destroy_cq(comm->cq);
        free(comm);
        return ncclSystemError;
    }

    comm->connected = 1;
    comm->remote_qp_num = connect_handle.qp_num;

    // TICKET-6: Add new connection to pool
    if (g_mesh_state.enable_conn_pool) {
        if (mesh_conn_pool_add(peer_ip, nic, comm->qp, comm->cq, comm->remote_qp_num) == 0) {
            // Find the entry we just added
            comm->pool_entry = mesh_conn_pool_find(peer_ip, nic);
        }
    }

    MESH_INFO("connect: Connected to peer %s via NIC %s (local QP %d -> remote QP %d)",
              peer_ip_str, nic->dev_name, comm->qp->qp_num, connect_handle.qp_num);


    *sendComm = comm;
    if (sendDevComm) *sendDevComm = NULL;
    return ncclSuccess;
}

static ncclResult_t mesh_accept(void *listenComm, void **recvComm,
                               ncclNetDeviceHandle_t **recvDevComm) {
    // Dispatch to TCP if in fallback mode (TICKET-4)
    if (g_mesh_state.tcp_fallback_active) {
        return mesh_tcp_accept_impl(listenComm, recvComm, recvDevComm);
    }

    struct mesh_listen_comm *lcomm = (struct mesh_listen_comm *)listenComm;
    struct mesh_recv_comm *rcomm;


    // Allocate recv comm
    rcomm = calloc(1, sizeof(*rcomm));
    if (!rcomm) {
        return ncclSystemError;
    }
    
    // Wait for handshake from queue (filled by handshake thread)
    pthread_mutex_lock(&lcomm->queue_mutex);
    
    // Wait with timeout for entry in queue
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    // Accept timeout is 6x configured timeout (default: 30s for 5000ms config)
    timeout.tv_sec += (g_mesh_state.timeout_ms / 1000) * 6;
    if (timeout.tv_sec < 5) timeout.tv_sec = 5;  // Minimum 5 second timeout
    
    while (lcomm->queue_head == lcomm->queue_tail) {
        int rc = pthread_cond_timedwait(&lcomm->queue_cond, &lcomm->queue_mutex, &timeout);
        if (rc == ETIMEDOUT) {
            pthread_mutex_unlock(&lcomm->queue_mutex);
            MESH_WARN("accept: Timeout waiting for handshake");
            free(rcomm);
            return ncclSystemError;
        }
    }
    
    // Get entry from queue
    struct handshake_entry *entry = &lcomm->handshake_queue[lcomm->queue_head];
    lcomm->queue_head = (lcomm->queue_head + 1) % HANDSHAKE_QUEUE_SIZE;
    
    // Copy data out
    rcomm->qp = entry->local_qp;
    rcomm->cq = entry->local_cq;
    rcomm->nic = entry->nic;
    entry->valid = 0;
    
    pthread_mutex_unlock(&lcomm->queue_mutex);
    
    
    rcomm->connected = 1;
    
    MESH_INFO("accept: Ready on %s (QP %d)", rcomm->nic->dev_name, rcomm->qp->qp_num);
    
    *recvComm = rcomm;
    if (recvDevComm) *recvDevComm = NULL;
    
    
    return ncclSuccess;
}

static ncclResult_t mesh_regMr(void *comm, void *data, size_t size, int type, void **mhandle) {
    // Dispatch to TCP if in fallback mode (TICKET-4)
    if (g_mesh_state.tcp_fallback_active) {
        return mesh_tcp_regMr(comm, data, size, type, mhandle);
    }

    struct mesh_send_comm *scomm = (struct mesh_send_comm *)comm;
    struct mesh_mr_handle *mrh;
    int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    MESH_DEBUG("regMr: comm=%p, data=%p, size=%zu, type=%d", comm, data, size, type);

    if (!scomm || !scomm->nic || !scomm->nic->pd) {
        MESH_WARN("regMr: invalid comm or nic");
        return ncclSystemError;
    }
    
    
    mrh = calloc(1, sizeof(*mrh));
    if (!mrh) {
        return ncclSystemError;
    }
    
    mrh->mr = ibv_reg_mr(scomm->nic->pd, data, size, access_flags);
    if (!mrh->mr) {
        MESH_WARN("Failed to register MR: %s", strerror(errno));
        free(mrh);
        return ncclSystemError;
    }
    
    
    mrh->nic = scomm->nic;
    mrh->addr = data;
    mrh->size = size;
    
    *mhandle = mrh;
    return ncclSuccess;
}

static ncclResult_t mesh_regMrDmaBuf(void *comm, void *data, size_t size, int type,
                                    uint64_t offset, int fd, void **mhandle) {
    // DMA-BUF not implemented yet
    return mesh_regMr(comm, data, size, type, mhandle);
}

static ncclResult_t mesh_deregMr(void *comm, void *mhandle) {
    struct mesh_mr_handle *mrh = (struct mesh_mr_handle *)mhandle;
    
    if (mrh && mrh->mr) {
        ibv_dereg_mr(mrh->mr);
    }
    free(mrh);
    
    return ncclSuccess;
}

static ncclResult_t mesh_isend(void *sendComm, void *data, int size, int tag,
                              void *mhandle, void **request) {
    // Dispatch to TCP if in fallback mode (TICKET-4)
    if (g_mesh_state.tcp_fallback_active) {
        return mesh_tcp_isend(sendComm, data, size, tag, mhandle, request);
    }

    struct mesh_send_comm *comm = (struct mesh_send_comm *)sendComm;
    struct mesh_mr_handle *mrh = (struct mesh_mr_handle *)mhandle;
    struct mesh_request *req;
    struct ibv_send_wr wr, *bad_wr;
    struct ibv_sge sge;

    (void)tag;

    MESH_DEBUG("isend: comm=%p, data=%p, size=%d", (void*)comm, data, size);

    if (!comm || !comm->qp) {
        MESH_WARN("isend: invalid comm");
        return ncclSystemError;
    }
    if (!mrh || !mrh->mr) {
        MESH_WARN("isend: invalid mhandle");
        return ncclSystemError;
    }

    // Fast-fail if peer already known to be disconnected
    if (comm->peer_failed) {
        MESH_WARN("isend: peer already failed (last_status=%d), failing fast",
                  comm->last_wc_status);
        return ncclSystemError;
    }

    req = calloc(1, sizeof(*req));
    if (!req) {
        return ncclSystemError;
    }
    __atomic_fetch_add(&g_mesh_state.requests_allocated, 1, __ATOMIC_RELAXED);

    req->used = 1;
    req->size = size;
    req->cq = comm->cq;  // Store CQ for polling
    req->done = 0;
    req->comm = comm;    // Track comm for error propagation
    req->is_send = 1;

    // Setup scatter/gather entry
    sge.addr = (uintptr_t)data;
    sge.length = size;
    sge.lkey = mrh->mr->lkey;

    // Setup send work request
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uintptr_t)req;
    wr.next = NULL;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;


    if (ibv_post_send(comm->qp, &wr, &bad_wr)) {
        MESH_WARN("Failed to post send: %s", strerror(errno));
        __atomic_fetch_add(&g_mesh_state.requests_freed, 1, __ATOMIC_RELAXED);
        free(req);
        return ncclSystemError;
    }

    // TICKET-9: Track request in comm for cleanup on close
    // This prevents memory leaks when operations hang and comm is closed
    if (comm->num_requests < MESH_MAX_QPS) {
        comm->requests[comm->num_requests++] = req;
    }

    *request = req;
    return ncclSuccess;
}

static ncclResult_t mesh_irecv(void *recvComm, int n, void **data, int *sizes,
                              int *tags, void **mhandles, void **request) {
    // Dispatch to TCP if in fallback mode (TICKET-4)
    if (g_mesh_state.tcp_fallback_active) {
        return mesh_tcp_irecv(recvComm, n, data, sizes, tags, mhandles, request);
    }

    struct mesh_recv_comm *comm = (struct mesh_recv_comm *)recvComm;
    struct mesh_request *req;
    struct ibv_recv_wr wr, *bad_wr;
    struct ibv_sge sge;

    (void)tags;

    MESH_DEBUG("irecv: comm=%p, n=%d, sizes[0]=%d", (void*)comm, n, sizes ? sizes[0] : 0);

    if (!comm || !comm->qp) {
        MESH_WARN("irecv: invalid comm");
        return ncclSystemError;
    }

    if (n != 1) {
        // For simplicity, only handle n=1 for now
        MESH_WARN("irecv with n=%d not supported yet", n);
        return ncclInternalError;
    }

    struct mesh_mr_handle *mrh = (struct mesh_mr_handle *)mhandles[0];

    if (!mrh || !mrh->mr) {
        MESH_WARN("irecv: invalid mhandle");
        return ncclSystemError;
    }

    // Fast-fail if peer already known to be disconnected
    if (comm->peer_failed) {
        MESH_WARN("irecv: peer already failed (last_status=%d), failing fast",
                  comm->last_wc_status);
        return ncclSystemError;
    }

    uint32_t lkey = mrh->mr->lkey;

    req = calloc(1, sizeof(*req));
    if (!req) {
        return ncclSystemError;
    }
    __atomic_fetch_add(&g_mesh_state.requests_allocated, 1, __ATOMIC_RELAXED);

    req->used = 1;
    req->size = sizes[0];
    req->cq = comm->cq;  // Store CQ for polling
    req->done = 0;
    req->comm = comm;    // Track comm for error propagation
    req->is_send = 0;

    // Setup scatter/gather entry
    sge.addr = (uintptr_t)data[0];
    sge.length = sizes[0];
    sge.lkey = lkey;

    // Setup receive work request
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uintptr_t)req;
    wr.next = NULL;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    if (ibv_post_recv(comm->qp, &wr, &bad_wr)) {
        MESH_WARN("Failed to post recv: %s", strerror(errno));
        __atomic_fetch_add(&g_mesh_state.requests_freed, 1, __ATOMIC_RELAXED);
        free(req);
        return ncclSystemError;
    }

    // TICKET-9: Track request in comm for cleanup on close
    // This prevents memory leaks when operations hang and comm is closed
    if (comm->num_requests < MESH_MAX_QPS) {
        comm->requests[comm->num_requests++] = req;
    }

    *request = req;
    return ncclSuccess;
}

static ncclResult_t mesh_iflush(void *recvComm, int n, void **data, int *sizes,
                               void **mhandles, void **request) {
    // No flush needed for verbs
    *request = NULL;
    return ncclSuccess;
}

/*
 * Check if a WC status indicates peer disconnection/failure
 * These errors typically mean the remote side is unreachable
 */
static int mesh_is_peer_failure(enum ibv_wc_status status) {
    switch (status) {
        case IBV_WC_RETRY_EXC_ERR:      // Transport retry counter exceeded
        case IBV_WC_RNR_RETRY_EXC_ERR:  // RNR retry counter exceeded
        case IBV_WC_REM_ABORT_ERR:      // Remote abort
        case IBV_WC_REM_ACCESS_ERR:     // Remote access error
        case IBV_WC_REM_INV_REQ_ERR:    // Remote invalid request
        case IBV_WC_REM_OP_ERR:         // Remote operation error
            return 1;
        default:
            return 0;
    }
}

/*
 * Mark peer as failed on a comm structure
 * This enables fast-fail for subsequent operations
 */
static void mesh_mark_peer_failed(struct mesh_request *req, enum ibv_wc_status status) {
    if (!req || !req->comm) return;

    if (req->is_send) {
        struct mesh_send_comm *comm = (struct mesh_send_comm *)req->comm;
        if (!comm->peer_failed) {
            comm->peer_failed = 1;
            comm->last_wc_status = status;
            MESH_WARN("Peer failure detected on send comm: status=%d (%s)",
                      status, ibv_wc_status_str(status));
        }
        comm->error_count++;
    } else {
        struct mesh_recv_comm *comm = (struct mesh_recv_comm *)req->comm;
        if (!comm->peer_failed) {
            comm->peer_failed = 1;
            comm->last_wc_status = status;
            MESH_WARN("Peer failure detected on recv comm: status=%d (%s)",
                      status, ibv_wc_status_str(status));
        }
        comm->error_count++;
    }
}

/*
 * TICKET-9: Remove request from comm's tracking array before freeing
 * This prevents dangling pointers in the comm->requests[] array
 */
static void mesh_untrack_request(struct mesh_request *req) {
    if (!req || !req->comm) return;

    if (req->is_send) {
        struct mesh_send_comm *comm = (struct mesh_send_comm *)req->comm;
        for (int i = 0; i < comm->num_requests; i++) {
            if (comm->requests[i] == req) {
                comm->requests[i] = NULL;
                return;
            }
        }
    } else {
        struct mesh_recv_comm *comm = (struct mesh_recv_comm *)req->comm;
        for (int i = 0; i < comm->num_requests; i++) {
            if (comm->requests[i] == req) {
                comm->requests[i] = NULL;
                return;
            }
        }
    }
}

static ncclResult_t mesh_test(void *request, int *done, int *sizes) {
    // Dispatch to TCP if in fallback mode (TICKET-4)
    if (g_mesh_state.tcp_fallback_active) {
        return mesh_tcp_test_impl(request, done, sizes);
    }

    struct mesh_request *req = (struct mesh_request *)request;
    struct ibv_wc wc;
    int ret;

    if (!req) {
        *done = 1;
        return ncclSuccess;
    }

    if (req->done) {
        *done = 1;
        if (sizes) *sizes = req->size;
        mesh_untrack_request(req);  // TICKET-9: Remove from comm tracking
        __atomic_fetch_add(&g_mesh_state.requests_freed, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED);
        free(req);  // TICKET-8: Free completed request to prevent memory leak
        return ncclSuccess;
    }

    if (!req->cq) {
        MESH_WARN("mesh_test: request has no CQ");
        req->done = 1;
        *done = 1;
        mesh_untrack_request(req);  // TICKET-9: Remove from comm tracking
        __atomic_fetch_add(&g_mesh_state.requests_freed, 1, __ATOMIC_RELAXED);
        free(req);  // TICKET-8: Free request on completion
        return ncclSuccess;
    }

    // Poll for completions - we might get completions for OTHER requests
    // that share this CQ, so keep polling until we find ours or CQ is empty
    while (1) {
        ret = ibv_poll_cq(req->cq, 1, &wc);
        if (ret < 0) {
            MESH_WARN("mesh_test: ibv_poll_cq failed: %s", strerror(errno));
            return ncclSystemError;
        }

        if (ret == 0) {
            // No more completions - our request is not done yet
            *done = 0;
            return ncclSuccess;
        }

        // Got a completion - get the associated request
        struct mesh_request *completed_req = (struct mesh_request *)(uintptr_t)wc.wr_id;

        // Check status
        if (wc.status != IBV_WC_SUCCESS) {
            // Log detailed error information
            MESH_WARN("mesh_test: WC error: status=%d (%s) vendor_err=0x%x opcode=%d",
                      wc.status, ibv_wc_status_str(wc.status), wc.vendor_err, wc.opcode);

            // Check if this indicates peer failure
            if (mesh_is_peer_failure(wc.status)) {
                mesh_mark_peer_failed(completed_req, wc.status);
            }

            // Mark the request as done (with error)
            if (completed_req) {
                completed_req->done = 1;
                completed_req->wc = wc;
            }

            // If this is our request, return error immediately
            if (completed_req == req) {
                *done = 1;
                mesh_untrack_request(req);  // TICKET-9: Remove from comm tracking
                __atomic_fetch_add(&g_mesh_state.requests_freed, 1, __ATOMIC_RELAXED);
                __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED);
                free(req);  // TICKET-8: Free request on error completion
                return ncclSystemError;
            }

            // Continue polling for other completions
            continue;
        }

        // Success - mark the request as done
        if (completed_req) {
            completed_req->done = 1;
            completed_req->wc = wc;
        }

        // Is it OUR request?
        if (completed_req == req) {
            *done = 1;
            if (sizes) *sizes = req->size;
            mesh_untrack_request(req);  // TICKET-9: Remove from comm tracking
            __atomic_fetch_add(&g_mesh_state.requests_freed, 1, __ATOMIC_RELAXED);
            uint64_t ops = __atomic_fetch_add(&g_mesh_state.ops_completed, 1, __ATOMIC_RELAXED) + 1;
            // TICKET-8: Periodic stats logging every 10000 ops
            if (ops % 10000 == 0) {
                uint64_t alloc = __atomic_load_n(&g_mesh_state.requests_allocated, __ATOMIC_RELAXED);
                uint64_t freed = __atomic_load_n(&g_mesh_state.requests_freed, __ATOMIC_RELAXED);
                MESH_INFO("Stats: ops=%lu alloc=%lu freed=%lu outstanding=%lu pool_hits=%lu pool_misses=%lu",
                          ops, alloc, freed, alloc - freed,
                          g_mesh_state.conn_pool.hits, g_mesh_state.conn_pool.misses);
            }
            free(req);  // TICKET-8: Free request on success completion
            return ncclSuccess;
        }

        // Not our request - keep polling
    }
}

static ncclResult_t mesh_closeSend(void *sendComm) {
    // Dispatch to TCP if in fallback mode (TICKET-4)
    if (g_mesh_state.tcp_fallback_active) {
        return mesh_tcp_closeSend(sendComm);
    }

    struct mesh_send_comm *comm = (struct mesh_send_comm *)sendComm;

    if (comm) {
        // TICKET-9: Free any outstanding requests to prevent memory leak
        // This handles the case where operations hang and comm is closed
        // before all operations complete (e.g., FSDP timeout during all-gather)
        for (int i = 0; i < comm->num_requests; i++) {
            struct mesh_request *req = comm->requests[i];
            if (req && !req->done) {
                MESH_DEBUG("closeSend: freeing outstanding request %d", i);
                __atomic_fetch_add(&g_mesh_state.requests_freed, 1, __ATOMIC_RELAXED);
                free(req);
            }
        }
        comm->num_requests = 0;

        // TICKET-6: Release pooled connection instead of destroying
        if (comm->pool_entry) {
            mesh_conn_pool_release(comm->pool_entry);
            // Don't destroy QP/CQ - they belong to the pool
        } else {
            // Not pooled - destroy resources
            if (comm->qp) ibv_destroy_qp(comm->qp);
            if (comm->cq) ibv_destroy_cq(comm->cq);
        }
        free(comm);
    }

    return ncclSuccess;
}

static ncclResult_t mesh_closeRecv(void *recvComm) {
    // Dispatch to TCP if in fallback mode (TICKET-4)
    if (g_mesh_state.tcp_fallback_active) {
        return mesh_tcp_closeRecv(recvComm);
    }

    struct mesh_recv_comm *comm = (struct mesh_recv_comm *)recvComm;

    if (comm) {
        // TICKET-9: Free any outstanding requests to prevent memory leak
        // This handles the case where operations hang and comm is closed
        // before all operations complete (e.g., FSDP timeout during all-gather)
        for (int i = 0; i < comm->num_requests; i++) {
            struct mesh_request *req = comm->requests[i];
            if (req && !req->done) {
                MESH_DEBUG("closeRecv: freeing outstanding request %d", i);
                __atomic_fetch_add(&g_mesh_state.requests_freed, 1, __ATOMIC_RELAXED);
                free(req);
            }
        }
        comm->num_requests = 0;

        // QP/CQ are now owned by recv_comm, destroy them
        if (comm->qp) ibv_destroy_qp(comm->qp);
        if (comm->cq) ibv_destroy_cq(comm->cq);
        free(comm);
    }

    return ncclSuccess;
}

static ncclResult_t mesh_closeListen(void *listenComm) {
    // Dispatch to TCP if in fallback mode (TICKET-4)
    if (g_mesh_state.tcp_fallback_active) {
        return mesh_tcp_closeListen(listenComm);
    }

    struct mesh_listen_comm *comm = (struct mesh_listen_comm *)listenComm;

    if (comm) {
        // Stop handshake thread
        if (comm->thread_running) {
            comm->thread_stop = 1;
            pthread_cond_broadcast(&comm->queue_cond);
            pthread_join(comm->handshake_thread, NULL);
            comm->thread_running = 0;
        }
        
        // Close handshake socket
        if (comm->handshake_sock >= 0) {
            close(comm->handshake_sock);
        }
        
        // Destroy mutex and condition
        pthread_mutex_destroy(&comm->queue_mutex);
        pthread_cond_destroy(&comm->queue_cond);
        
        // Clean up any remaining queue entries
        for (int i = 0; i < HANDSHAKE_QUEUE_SIZE; i++) {
            if (comm->handshake_queue[i].valid) {
                if (comm->handshake_queue[i].local_qp) 
                    ibv_destroy_qp(comm->handshake_queue[i].local_qp);
                if (comm->handshake_queue[i].local_cq)
                    ibv_destroy_cq(comm->handshake_queue[i].local_cq);
            }
        }
        
        for (int i = 0; i < comm->num_qps; i++) {
            if (comm->qps[i].qp) ibv_destroy_qp(comm->qps[i].qp);
            if (comm->qps[i].cq) ibv_destroy_cq(comm->qps[i].cq);
        }
        free(comm);
    }
    
    return ncclSuccess;
}

static ncclResult_t mesh_getDeviceMr(void *comm, void *mhandle, void **dptr_mhandle) {
    *dptr_mhandle = NULL;
    return ncclSuccess;
}

static ncclResult_t mesh_irecvConsumed(void *recvComm, int n, void *request) {
    return ncclSuccess;
}

/*
 * ============================================================================
 * Plugin Export
 * ============================================================================
 */

__attribute__((visibility("default")))
const ncclNet_v8_t ncclNetPlugin_v8 = {
    .name = PLUGIN_NAME,
    .init = mesh_init,
    .devices = mesh_devices,
    .getProperties = mesh_getProperties,
    .listen = mesh_listen,
    .connect = mesh_connect,
    .accept = mesh_accept,
    .regMr = mesh_regMr,
    .regMrDmaBuf = mesh_regMrDmaBuf,
    .deregMr = mesh_deregMr,
    .isend = mesh_isend,
    .irecv = mesh_irecv,
    .iflush = mesh_iflush,
    .test = mesh_test,
    .closeSend = mesh_closeSend,
    .closeRecv = mesh_closeRecv,
    .closeListen = mesh_closeListen,
    .getDeviceMr = mesh_getDeviceMr,
    .irecvConsumed = mesh_irecvConsumed,
};

// Alias for NCCL to find
__attribute__((visibility("default")))
const ncclNet_v8_t *ncclNet_v8 = &ncclNetPlugin_v8;

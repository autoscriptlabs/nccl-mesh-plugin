/*
 * NCCL Mesh Plugin - Subnet-aware RDMA transport
 * 
 * Enables NCCL to work with direct-connect mesh topologies where
 * each node pair is on a different subnet.
 */

#ifndef NCCL_MESH_PLUGIN_H
#define NCCL_MESH_PLUGIN_H

#include <stdint.h>
#include <pthread.h>
#include <infiniband/verbs.h>

#define MESH_MAX_NICS 8
#define MESH_MAX_QPS 256
#define MESH_MAX_MRS 1024
#define MESH_HANDLE_MAGIC 0x4D455348  // "MESH"

// Forward declarations
struct mesh_plugin_state;
struct mesh_nic;
struct mesh_comm;

/*
 * Represents one RDMA-capable NIC with its subnet information
 */
struct mesh_nic {
    // RDMA resources
    struct ibv_context *context;
    struct ibv_pd *pd;
    int port_num;
    int gid_index;
    
    // Network addressing
    uint32_t ip_addr;           // Host byte order
    uint32_t netmask;           // Host byte order  
    uint32_t subnet;            // ip_addr & netmask
    
    // Device identification
    char dev_name[64];          // RDMA device name (e.g., "rocep1s0f1")
    char if_name[64];           // Network interface name (e.g., "enp1s0f1np1")
    char pci_path[256];         // PCI bus path
    
    // Capabilities
    int max_qp;
    int max_cq;
    int max_mr;
    int max_sge;
    uint64_t max_mr_size;
    int gdr_supported;          // GPUDirect RDMA support
    
    // Statistics
    uint64_t bytes_sent;
    uint64_t bytes_recv;
    uint64_t connections;
};

/*
 * Address entry for multi-homed hosts
 */
#define MESH_MAX_ADDRS 6

struct mesh_addr_entry {
    uint32_t ip;                // IP address (network byte order)
    uint32_t mask;              // Subnet mask (network byte order)
    uint16_t qp_num;            // QP number for this NIC
    uint8_t  nic_idx;           // Index into our NIC array
    uint8_t  gid_index;         // GID index for this NIC
};

/*
 * Connection handle - exchanged between peers during setup
 * Must fit within NCCL_NET_HANDLE_MAXSIZE (128 bytes)
 */
struct mesh_handle {
    uint32_t magic;             // MESH_HANDLE_MAGIC
    uint8_t  num_addrs;         // Number of valid addresses
    uint8_t  selected_idx;      // Which address was selected (set by connect)
    uint16_t lid;               // IB LID (0 for RoCE)
    uint16_t qp_num;            // QP number (for compat with mesh_connect_qp)
    uint16_t handshake_port;    // TCP port for QP handshake
    uint8_t  port_num;          // Port number (usually 1)
    uint8_t  mtu;               // MTU setting
    uint32_t psn;               // Packet sequence number
    uint32_t handshake_ip;      // IP address for handshake (network byte order)
    union ibv_gid gid;          // GID (16 bytes)
    struct mesh_addr_entry addrs[MESH_MAX_ADDRS];  // 12 bytes each
    // Total: 4+1+1+2+2+2+1+1+4+4+16 + 6*12 = 38 + 72 = 110 bytes (fits in 128)
};

/*
 * Listen state - waiting for incoming connections
 * Creates QPs on ALL NICs so any peer can connect
 */
#define HANDSHAKE_QUEUE_SIZE 16

/*
 * QP info exchanged during handshake
 */
struct mesh_qp_info {
    uint32_t qp_num;       // Network byte order
    uint32_t psn;          // Network byte order
    uint8_t gid[16];       // Raw GID
    uint32_t ip;           // Network byte order
    uint8_t gid_index;
    uint8_t nic_idx;       // Which NIC on the listener
    uint8_t reserved[2];
};

struct handshake_entry {
    struct mesh_qp_info remote_info;
    struct ibv_qp *local_qp;
    struct ibv_cq *local_cq;
    struct mesh_nic *nic;
    int valid;
};

struct mesh_listen_comm {
    int num_qps;
    struct {
        struct mesh_nic *nic;
        struct ibv_qp *qp;
        struct ibv_cq *cq;
    } qps[MESH_MAX_NICS];
    uint32_t psn;
    int ready;
    
    // Handshake socket for QP info exchange
    int handshake_sock;
    uint16_t handshake_port;
    uint32_t handshake_ip;
    
    // Background handshake thread
    pthread_t handshake_thread;
    int thread_running;
    int thread_stop;
    
    // Queue of received handshakes for accept() to consume
    struct handshake_entry handshake_queue[HANDSHAKE_QUEUE_SIZE];
    int queue_head;
    int queue_tail;
    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_cond;
};

/*
 * Send/Receive communication state
 */
struct mesh_send_comm {
    struct mesh_nic *nic;
    struct ibv_qp *qp;
    struct ibv_cq *cq;
    uint32_t remote_qp_num;
    union ibv_gid remote_gid;
    int connected;

    // Peer health tracking
    int peer_failed;            // Set when peer disconnect detected
    int last_wc_status;         // Last WC error status (for diagnostics)
    uint64_t error_count;       // Number of errors seen

    // Request tracking
    struct mesh_request *requests[MESH_MAX_QPS];
    int num_requests;
};

struct mesh_recv_comm {
    struct mesh_nic *nic;
    struct ibv_qp *qp;
    struct ibv_cq *cq;
    int connected;

    // Peer health tracking
    int peer_failed;            // Set when peer disconnect detected
    int last_wc_status;         // Last WC error status (for diagnostics)
    uint64_t error_count;       // Number of errors seen

    // Request tracking
    struct mesh_request *requests[MESH_MAX_QPS];
    int num_requests;
};

/*
 * Memory registration handle
 */
struct mesh_mr_handle {
    struct ibv_mr *mr;
    struct mesh_nic *nic;
    void *addr;
    size_t size;
};

/*
 * Async request state
 */
struct mesh_request {
    int used;
    int done;
    size_t size;
    struct ibv_cq *cq;          // CQ to poll for completion
    struct ibv_wc wc;
    void *comm;                 // Associated send/recv comm (for error propagation)
    int is_send;                // 1 if send request, 0 if recv
};

/*
 * Global plugin state
 */
struct mesh_plugin_state {
    struct mesh_nic nics[MESH_MAX_NICS];
    int num_nics;
    int initialized;

    // Configuration
    int gid_index;              // From NCCL_MESH_GID_INDEX
    int debug;                  // From NCCL_MESH_DEBUG
    int fast_fail;              // From NCCL_MESH_FAST_FAIL: reduce retries for faster failure detection

    // Logging (provided by NCCL)
    void (*log_fn)(int level, unsigned long flags, const char *file,
                   int line, const char *fmt, ...);
};

// Global state (singleton)
extern struct mesh_plugin_state g_mesh_state;

/*
 * Internal functions
 */

// Initialization
int mesh_init_nics(void);
int mesh_discover_nic_ips(void);
int mesh_setup_nic(struct mesh_nic *nic, struct ibv_device *device);

// Routing
struct mesh_nic* mesh_find_nic_for_ip(uint32_t peer_ip);
struct mesh_nic* mesh_find_nic_by_name(const char *name);
int mesh_get_nic_index(struct mesh_nic *nic);

// RDMA operations
int mesh_create_qp(struct mesh_nic *nic, struct ibv_qp **qp, struct ibv_cq **cq);
int mesh_connect_qp(struct ibv_qp *qp, struct mesh_nic *nic, struct mesh_handle *remote);
int mesh_post_send(struct mesh_send_comm *comm, void *data, size_t size, 
                   struct mesh_mr_handle *mr, struct mesh_request *req);
int mesh_post_recv(struct mesh_recv_comm *comm, void *data, size_t size,
                   struct mesh_mr_handle *mr, struct mesh_request *req);
int mesh_poll_cq(struct ibv_cq *cq, struct mesh_request *req);

// Utilities
uint32_t mesh_ip_to_uint(const char *ip_str);
void mesh_uint_to_ip(uint32_t ip, char *buf, size_t len);
int mesh_get_interface_ip(const char *if_name, uint32_t *ip, uint32_t *mask);
const char* mesh_find_netdev_for_rdma(const char *rdma_dev);

// Logging macros
#define MESH_LOG(level, fmt, ...) \
    do { \
        if (g_mesh_state.log_fn) { \
            g_mesh_state.log_fn(level, 0, __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#define MESH_INFO(fmt, ...) MESH_LOG(NCCL_LOG_INFO, "MESH " fmt, ##__VA_ARGS__)
#define MESH_WARN(fmt, ...) MESH_LOG(NCCL_LOG_WARN, "MESH " fmt, ##__VA_ARGS__)
#define MESH_DEBUG(fmt, ...) \
    do { if (g_mesh_state.debug) MESH_LOG(NCCL_LOG_TRACE, "MESH " fmt, ##__VA_ARGS__); } while(0)

#endif // NCCL_MESH_PLUGIN_H

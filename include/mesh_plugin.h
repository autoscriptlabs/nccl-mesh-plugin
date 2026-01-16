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

// Connection pool settings (TICKET-6)
#define MESH_CONN_POOL_SIZE 64        // Max pooled connections per NIC
#define MESH_ASYNC_QUEUE_SIZE 32      // Pending async connect requests

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

    // Cached GID (TICKET-5: avoid repeated queries and handle changes gracefully)
    union ibv_gid cached_gid;   // Cached GID value from initialization
    int gid_valid;              // 1 if cached_gid is valid, 0 otherwise
    uint64_t gid_query_time;    // Timestamp of last GID query (for staleness check)

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
    enum ibv_mtu active_mtu;    // Port's active MTU (queried from hardware)

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

    // Connection pooling (TICKET-6)
    struct mesh_conn_pool_entry *pool_entry;  // Pooled connection (if using pool)
    uint32_t peer_ip;           // Peer IP for pool lookup

    // Async connect (TICKET-7)
    struct mesh_async_connect_req *async_req; // Pending async connect request
    int connect_pending;        // 1 if waiting for async connect to complete

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
    int is_tcp;                 // 1 if this is a TCP fallback registration
};

/*
 * TCP fallback communication structures (TICKET-4)
 * Used when RDMA/IB setup fails or is disabled
 */
struct mesh_tcp_listen_comm {
    int listen_sock;            // TCP listening socket
    uint16_t listen_port;       // Port we're listening on
    uint32_t listen_ip;         // IP we're bound to (INADDR_ANY typically)
    int ready;
};

struct mesh_tcp_send_comm {
    int sock;                   // Connected TCP socket
    uint32_t remote_ip;         // Remote peer IP
    uint16_t remote_port;       // Remote peer port
    int connected;

    // Peer health tracking
    int peer_failed;
    int last_errno;
    uint64_t error_count;

    // Buffer for message framing
    uint8_t send_hdr[8];        // Size header for message framing
};

struct mesh_tcp_recv_comm {
    int sock;                   // Connected TCP socket
    uint32_t remote_ip;         // Remote peer IP
    int connected;

    // Peer health tracking
    int peer_failed;
    int last_errno;
    uint64_t error_count;

    // Buffer for message framing
    uint8_t recv_hdr[8];        // Size header for message framing
};

struct mesh_tcp_request {
    int used;
    int done;
    size_t size;
    void *data;                 // Buffer for async completion
    int is_send;                // 1 if send, 0 if recv
    void *comm;                 // Associated comm
    int error;                  // Error code if failed
    size_t offset;              // TICKET-10: Bytes sent/received so far for async progress
    int header_sent;            // TICKET-10: 1 if size header already sent (send only)
};

/*
 * Connection pool entry (TICKET-6)
 * Represents a reusable QP connection to a specific peer
 */
struct mesh_conn_pool_entry {
    int in_use;                 // 1 if currently assigned to a comm
    int valid;                  // 1 if QP is connected and usable
    uint32_t peer_ip;           // Remote peer IP (host byte order)
    uint32_t remote_qp_num;     // Remote QP number
    struct mesh_nic *nic;       // Local NIC
    struct ibv_qp *qp;          // Queue pair
    struct ibv_cq *cq;          // Completion queue
    uint64_t last_used;         // Timestamp of last use (for LRU eviction)
    int ref_count;              // Number of comms using this connection
};

/*
 * Connection pool (TICKET-6)
 * Pools QP connections for reuse between same node pairs
 */
struct mesh_conn_pool {
    struct mesh_conn_pool_entry entries[MESH_CONN_POOL_SIZE];
    int num_entries;
    pthread_mutex_t mutex;      // Protects pool access
    uint64_t hits;              // Cache hits
    uint64_t misses;            // Cache misses
};

/*
 * Async connect request (TICKET-7)
 * Queued for background connection establishment
 */
struct mesh_async_connect_req {
    int valid;                  // 1 if this slot is in use
    int complete;               // 1 if connection is established
    int error;                  // Error code if failed
    struct mesh_handle handle;  // Copy of peer handle
    struct mesh_nic *nic;       // Selected NIC
    struct mesh_addr_entry *selected_addr;  // Selected peer address
    uint32_t peer_ip;           // Peer IP for lookup
    struct ibv_qp *qp;          // Created QP (set when complete)
    struct ibv_cq *cq;          // Created CQ (set when complete)
    uint32_t remote_qp_num;     // Remote QP number (set when complete)
    void *send_comm;            // Associated send_comm waiting for this
};

/*
 * Async connect state (TICKET-7)
 * Background thread for non-blocking connection establishment
 */
struct mesh_async_connect_state {
    pthread_t thread;           // Background connect thread
    int thread_running;         // 1 if thread is active
    int thread_stop;            // 1 to signal thread to stop
    struct mesh_async_connect_req queue[MESH_ASYNC_QUEUE_SIZE];
    int queue_head;             // Next slot to consume
    int queue_tail;             // Next slot to produce
    pthread_mutex_t mutex;      // Protects queue
    pthread_cond_t cond;        // Signals new work available
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

    // Configuration from environment variables
    int gid_index;              // NCCL_MESH_GID_INDEX: RoCE GID index (default: 3)
    int debug_level;            // NCCL_MESH_DEBUG: 0=off, 1=info, 2=verbose (default: 0)
    int fast_fail;              // NCCL_MESH_FAST_FAIL: reduce retries for faster failure detection
    int timeout_ms;             // NCCL_MESH_TIMEOUT_MS: connection timeout in ms (default: 5000)
    int retry_count;            // NCCL_MESH_RETRY_COUNT: retry attempts (default: 3)
    int disable_rdma;           // NCCL_MESH_DISABLE_RDMA: force TCP fallback

    // Connection pooling config (TICKET-6)
    int enable_conn_pool;       // NCCL_MESH_CONN_POOL: enable connection pooling (default: 1)

    // Async connect config (TICKET-7)
    int enable_async_connect;   // NCCL_MESH_ASYNC_CONNECT: enable async connect (default: 1)

    // TCP fallback state (TICKET-4)
    int tcp_fallback_active;    // 1 if using TCP fallback, 0 if RDMA
    int rdma_init_failed;       // 1 if RDMA init failed (used to trigger fallback)

    // Connection pool (TICKET-6)
    struct mesh_conn_pool conn_pool;

    // Async connect state (TICKET-7)
    struct mesh_async_connect_state async_connect;

    // TICKET-8: Request tracking for leak detection
    uint64_t requests_allocated;        // Total requests allocated
    uint64_t requests_freed;            // Total requests freed
    uint64_t tcp_requests_allocated;    // TCP requests allocated
    uint64_t tcp_requests_freed;        // TCP requests freed
    uint64_t ops_completed;             // Total send/recv operations completed

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

// GID management (TICKET-5: cache and validate GID)
int mesh_cache_gid(struct mesh_nic *nic);
int mesh_validate_gid(struct mesh_nic *nic);
int mesh_get_gid(struct mesh_nic *nic, union ibv_gid *gid);

// Connection pool (TICKET-6)
int mesh_conn_pool_init(void);
void mesh_conn_pool_destroy(void);
struct mesh_conn_pool_entry* mesh_conn_pool_acquire(uint32_t peer_ip, struct mesh_nic *nic);
void mesh_conn_pool_release(struct mesh_conn_pool_entry *entry);
struct mesh_conn_pool_entry* mesh_conn_pool_find(uint32_t peer_ip, struct mesh_nic *nic);
int mesh_conn_pool_add(uint32_t peer_ip, struct mesh_nic *nic,
                       struct ibv_qp *qp, struct ibv_cq *cq, uint32_t remote_qp_num);

// Async connect (TICKET-7)
int mesh_async_connect_init(void);
void mesh_async_connect_destroy(void);
struct mesh_async_connect_req* mesh_async_connect_submit(struct mesh_handle *handle,
                                                          struct mesh_nic *nic,
                                                          struct mesh_addr_entry *addr,
                                                          void *send_comm);
int mesh_async_connect_poll(struct mesh_async_connect_req *req);

// TCP fallback operations (TICKET-4)
int mesh_tcp_init(void);
int mesh_tcp_listen(int dev, void *handle, void **listenComm);
int mesh_tcp_connect(int dev, void *handle, void **sendComm);
int mesh_tcp_accept(void *listenComm, void **recvComm);
int mesh_tcp_send(void *sendComm, void *data, size_t size, void **request);
int mesh_tcp_recv(void *recvComm, void *data, size_t size, void **request);
int mesh_tcp_test(void *request, int *done, int *sizes);
int mesh_tcp_close_send(void *sendComm);
int mesh_tcp_close_recv(void *recvComm);
int mesh_tcp_close_listen(void *listenComm);

// Utilities
uint32_t mesh_ip_to_uint(const char *ip_str);
void mesh_uint_to_ip(uint32_t ip, char *buf, size_t len);
int mesh_get_interface_ip(const char *if_name, uint32_t *ip, uint32_t *mask);
const char* mesh_find_netdev_for_rdma(const char *rdma_dev);

// Logging macros
// NCCL_MESH_DEBUG levels: 0=off, 1=info/errors, 2=verbose/trace
#define MESH_LOG(level, fmt, ...) \
    do { \
        if (g_mesh_state.log_fn) { \
            g_mesh_state.log_fn(level, 0, __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
        } \
    } while(0)

// MESH_WARN: Always logged (errors and warnings)
#define MESH_WARN(fmt, ...) MESH_LOG(NCCL_LOG_WARN, "MESH " fmt, ##__VA_ARGS__)

// MESH_INFO: Logged at debug_level >= 1 (informational messages)
#define MESH_INFO(fmt, ...) \
    do { if (g_mesh_state.debug_level >= 1) MESH_LOG(NCCL_LOG_INFO, "MESH " fmt, ##__VA_ARGS__); } while(0)

// MESH_DEBUG: Logged at debug_level >= 2 (verbose/trace messages)
#define MESH_DEBUG(fmt, ...) \
    do { if (g_mesh_state.debug_level >= 2) MESH_LOG(NCCL_LOG_TRACE, "MESH " fmt, ##__VA_ARGS__); } while(0)

// MESH_TRACE: Alias for MESH_DEBUG (very verbose tracing)
#define MESH_TRACE(fmt, ...) MESH_DEBUG(fmt, ##__VA_ARGS__)

#endif // NCCL_MESH_PLUGIN_H

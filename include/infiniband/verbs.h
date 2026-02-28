/*
 * Comprehensive stub infiniband/verbs.h for compilation without libibverbs
 */
#ifndef INFINIBAND_VERBS_H
#define INFINIBAND_VERBS_H

#include <stdint.h>
#include <stddef.h>

/* MTU enums */
enum ibv_mtu {
    IBV_MTU_256 = 1,
    IBV_MTU_512 = 2,
    IBV_MTU_1024 = 3,
    IBV_MTU_2048 = 4,
    IBV_MTU_4096 = 5
};

/* Work completion status */
enum ibv_wc_status {
    IBV_WC_SUCCESS = 0,
    IBV_WC_LOC_LEN_ERR = 1,
    IBV_WC_LOC_QP_OP_ERR = 2,
    IBV_WC_LOC_EEC_OP_ERR = 3,
    IBV_WC_LOC_PROT_ERR = 4,
    IBV_WC_WR_FLUSH_ERR = 5,
    IBV_WC_MW_BIND_ERR = 6,
    IBV_WC_BAD_RESP_ERR = 7,
    IBV_WC_LOC_ACCESS_ERR = 8,
    IBV_WC_REM_INV_REQ_ERR = 9,
    IBV_WC_REM_ACCESS_ERR = 10,
    IBV_WC_REM_OP_ERR = 11,
    IBV_WC_RETRY_EXC_ERR = 12,
    IBV_WC_RNR_RETRY_EXC_ERR = 13,
    IBV_WC_LOC_RDD_VIOL_ERR = 14,
    IBV_WC_REM_INV_RD_REQ_ERR = 15,
    IBV_WC_REM_ABORT_ERR = 16,
    IBV_WC_INV_EECN_ERR = 17,
    IBV_WC_INV_EEC_STATE_ERR = 18,
    IBV_WC_FATAL_ERR = 19,
    IBV_WC_RESP_TIMEOUT_ERR = 20,
    IBV_WC_GENERAL_ERR = 21
};

/* Work completion opcode */
enum ibv_wc_opcode {
    IBV_WC_SEND = 0,
    IBV_WC_RDMA_WRITE = 1,
    IBV_WC_RDMA_READ = 2,
    IBV_WC_COMP_SWAP = 3,
    IBV_WC_FETCH_ADD = 4,
    IBV_WC_BIND_MW = 5,
    IBV_WC_RECV = 128,
    IBV_WC_RECV_RDMA_WITH_IMM = 129
};

/* QP type */
enum ibv_qp_type {
    IBV_QPT_RC = 2,
    IBV_QPT_UC = 3,
    IBV_QPT_UD = 4
};

/* QP state */
enum ibv_qp_state {
    IBV_QPS_RESET = 0,
    IBV_QPS_INIT = 1,
    IBV_QPS_RTR = 2,
    IBV_QPS_RTS = 3,
    IBV_QPS_SQD = 4,
    IBV_QPS_SQE = 5,
    IBV_QPS_ERR = 6
};

/* QP attribute mask */
enum ibv_qp_attr_mask {
    IBV_QP_STATE = 1 << 0,
    IBV_QP_CUR_STATE = 1 << 1,
    IBV_QP_EN_SQD_ASYNC_NOTIFY = 1 << 2,
    IBV_QP_ACCESS_FLAGS = 1 << 3,
    IBV_QP_PKEY_INDEX = 1 << 4,
    IBV_QP_PORT = 1 << 5,
    IBV_QP_QKEY = 1 << 6,
    IBV_QP_AV = 1 << 7,
    IBV_QP_PATH_MTU = 1 << 8,
    IBV_QP_TIMEOUT = 1 << 9,
    IBV_QP_RETRY_CNT = 1 << 10,
    IBV_QP_RNR_RETRY = 1 << 11,
    IBV_QP_RQ_PSN = 1 << 12,
    IBV_QP_MAX_QP_RD_ATOMIC = 1 << 13,
    IBV_QP_ALT_PATH = 1 << 14,
    IBV_QP_MIN_RNR_TIMER = 1 << 15,
    IBV_QP_SQ_PSN = 1 << 16,
    IBV_QP_MAX_DEST_RD_ATOMIC = 1 << 17,
    IBV_QP_PATH_MIG_STATE = 1 << 18,
    IBV_QP_CAP = 1 << 19,
    IBV_QP_DEST_QPN = 1 << 20
};

/* Send flags */
enum ibv_send_flags {
    IBV_SEND_FENCE = 1 << 0,
    IBV_SEND_SIGNALED = 1 << 1,
    IBV_SEND_SOLICITED = 1 << 2,
    IBV_SEND_INLINE = 1 << 3
};

/* WR opcode */
enum ibv_wr_opcode {
    IBV_WR_RDMA_WRITE = 0,
    IBV_WR_RDMA_WRITE_WITH_IMM = 1,
    IBV_WR_SEND = 2,
    IBV_WR_SEND_WITH_IMM = 3,
    IBV_WR_RDMA_READ = 4,
    IBV_WR_ATOMIC_CMP_AND_SWP = 5,
    IBV_WR_ATOMIC_FETCH_AND_ADD = 6
};

/* GID type */
union ibv_gid {
    uint8_t raw[16];
    struct {
        uint64_t subnet_prefix;
        uint64_t interface_id;
    } global;
};

/* Memory registration access flags */
enum ibv_access_flags {
    IBV_ACCESS_LOCAL_WRITE = 1,
    IBV_ACCESS_REMOTE_WRITE = 2,
    IBV_ACCESS_REMOTE_READ = 4,
    IBV_ACCESS_REMOTE_ATOMIC = 8
};

/* Device attributes */
struct ibv_device_attr {
    char fw_ver[64];
    uint64_t node_guid;
    uint64_t sys_image_guid;
    uint64_t max_mr_size;
    uint64_t page_size_cap;
    uint32_t vendor_id;
    uint32_t vendor_part_id;
    uint32_t hw_ver;
    int max_qp;
    int max_qp_wr;
    int device_cap_flags;
    int max_sge;
    int max_sge_rd;
    int max_cq;
    int max_cqe;
    int max_mr;
    int max_pd;
    int max_qp_rd_atom;
    int max_ee_rd_atom;
    int max_res_rd_atom;
    int max_qp_init_rd_atom;
    int max_ee_init_rd_atom;
    int atomic_cap;
    int max_ee;
    int max_rdd;
    int max_mw;
    int max_raw_ipv6_qp;
    int max_raw_ethy_qp;
    int max_mcast_grp;
    int max_mcast_qp_attach;
    int max_total_mcast_qp_attach;
    int max_ah;
    int max_fmr;
    int max_map_per_fmr;
    int max_srq;
    int max_srq_wr;
    int max_srq_sge;
    uint16_t max_pkeys;
    uint8_t local_ca_ack_delay;
    uint8_t phys_port_cnt;
};

/* Port attributes */
struct ibv_port_attr {
    enum ibv_mtu active_mtu;
    enum ibv_mtu max_mtu;
    uint32_t state;
    uint32_t phys_state;
    int gid_tbl_len;
    uint32_t port_cap_flags;
    uint32_t max_msg_sz;
    uint32_t bad_pkey_cntr;
    uint32_t qkey_viol_cntr;
    uint16_t pkey_tbl_len;
    uint16_t lid;
    uint16_t sm_lid;
    uint8_t lmc;
    uint8_t max_vl_num;
    uint8_t sm_sl;
    uint8_t subnet_timeout;
    uint8_t init_type_reply;
    uint8_t active_width;
    uint8_t active_speed;
    uint8_t link_layer;
};

/* Global route header */
struct ibv_global_route {
    union ibv_gid dgid;
    uint32_t flow_label;
    uint8_t sgid_index;
    uint8_t hop_limit;
    uint8_t traffic_class;
};

/* Address handle attributes */
struct ibv_ah_attr {
    struct ibv_global_route grh;
    uint16_t dlid;
    uint8_t sl;
    uint8_t src_path_bits;
    uint8_t static_rate;
    uint8_t is_global;
    uint8_t port_num;
};

/* QP capabilities */
struct ibv_qp_cap {
    uint32_t max_send_wr;
    uint32_t max_recv_wr;
    uint32_t max_send_sge;
    uint32_t max_recv_sge;
    uint32_t max_inline_data;
};

/* QP init attributes */
struct ibv_qp_init_attr {
    void *qp_context;
    struct ibv_cq *send_cq;
    struct ibv_cq *recv_cq;
    struct ibv_srq *srq;
    struct ibv_qp_cap cap;
    enum ibv_qp_type qp_type;
    int sq_sig_all;
};

/* QP attributes */
struct ibv_qp_attr {
    enum ibv_qp_state qp_state;
    enum ibv_qp_state cur_qp_state;
    enum ibv_mtu path_mtu;
    uint32_t path_mig_state;
    uint32_t qkey;
    uint32_t rq_psn;
    uint32_t sq_psn;
    uint32_t dest_qp_num;
    int qp_access_flags;
    struct ibv_qp_cap cap;
    struct ibv_ah_attr ah_attr;
    struct ibv_ah_attr alt_ah_attr;
    uint16_t pkey_index;
    uint16_t alt_pkey_index;
    uint8_t en_sqd_async_notify;
    uint8_t sq_draining;
    uint8_t max_rd_atomic;
    uint8_t max_dest_rd_atomic;
    uint8_t min_rnr_timer;
    uint8_t port_num;
    uint8_t timeout;
    uint8_t retry_cnt;
    uint8_t rnr_retry;
    uint8_t alt_port_num;
    uint8_t alt_timeout;
};

/* Work completion */
struct ibv_wc {
    uint64_t wr_id;
    enum ibv_wc_status status;
    enum ibv_wc_opcode opcode;
    uint32_t vendor_err;
    uint32_t byte_len;
    uint32_t imm_data;
    uint32_t qp_num;
    uint32_t src_qp;
    int wc_flags;
    uint16_t pkey_index;
    uint16_t slid;
    uint8_t sl;
    uint8_t dlid_path_bits;
};

/* Scatter/gather element */
struct ibv_sge {
    uint64_t addr;
    uint32_t length;
    uint32_t lkey;
};

/* Send work request */
struct ibv_send_wr {
    uint64_t wr_id;
    struct ibv_send_wr *next;
    struct ibv_sge *sg_list;
    int num_sge;
    enum ibv_wr_opcode opcode;
    int send_flags;
    uint32_t imm_data;
    union {
        struct {
            uint64_t remote_addr;
            uint32_t rkey;
        } rdma;
        struct {
            uint64_t remote_addr;
            uint64_t compare_add;
            uint64_t swap;
            uint32_t rkey;
        } atomic;
        struct {
            struct ibv_ah *ah;
            uint32_t remote_qpn;
            uint32_t remote_qkey;
        } ud;
    } wr;
};

/* Receive work request */
struct ibv_recv_wr {
    uint64_t wr_id;
    struct ibv_recv_wr *next;
    struct ibv_sge *sg_list;
    int num_sge;
};

/* Async event types (NCCL-001: for event monitoring) */
enum ibv_event_type {
    IBV_EVENT_CQ_ERR = 0,
    IBV_EVENT_QP_FATAL = 1,
    IBV_EVENT_QP_REQ_ERR = 2,
    IBV_EVENT_QP_ACCESS_ERR = 3,
    IBV_EVENT_COMM_EST = 4,
    IBV_EVENT_SQ_DRAINED = 5,
    IBV_EVENT_PATH_MIG = 6,
    IBV_EVENT_PATH_MIG_ERR = 7,
    IBV_EVENT_DEVICE_FATAL = 8,
    IBV_EVENT_PORT_ACTIVE = 9,
    IBV_EVENT_PORT_ERR = 10,
    IBV_EVENT_LID_CHANGE = 11,
    IBV_EVENT_PKEY_CHANGE = 12,
    IBV_EVENT_SM_CHANGE = 13,
    IBV_EVENT_SRQ_ERR = 14,
    IBV_EVENT_SRQ_LIMIT_REACHED = 15,
    IBV_EVENT_QP_LAST_WQE_REACHED = 16,
    IBV_EVENT_CLIENT_REREGISTER = 17,
    IBV_EVENT_GID_CHANGE = 18
};

/* Async event structure */
struct ibv_async_event {
    union {
        struct ibv_cq *cq;
        struct ibv_qp *qp;
        struct ibv_srq *srq;
        int port_num;
    } element;
    enum ibv_event_type event_type;
};

/* Opaque structs */
struct ibv_context {
    struct ibv_device *device;
    int cmd_fd;
    int async_fd;
    int num_comp_vectors;
};

struct ibv_device {
    char name[64];
    char dev_name[64];
    char dev_path[256];
    char ibdev_path[256];
};

struct ibv_pd {
    struct ibv_context *context;
    uint32_t handle;
};

struct ibv_mr {
    struct ibv_context *context;
    struct ibv_pd *pd;
    void *addr;
    size_t length;
    uint32_t handle;
    uint32_t lkey;
    uint32_t rkey;
};

struct ibv_cq {
    struct ibv_context *context;
    void *channel;
    void *cq_context;
    uint32_t handle;
    int cqe;
};

struct ibv_qp {
    struct ibv_context *context;
    void *qp_context;
    struct ibv_pd *pd;
    struct ibv_cq *send_cq;
    struct ibv_cq *recv_cq;
    struct ibv_srq *srq;
    uint32_t handle;
    uint32_t qp_num;
    enum ibv_qp_state state;
    enum ibv_qp_type qp_type;
};

struct ibv_srq {
    struct ibv_context *context;
    void *srq_context;
    struct ibv_pd *pd;
    uint32_t handle;
};

struct ibv_ah {
    struct ibv_context *context;
    struct ibv_pd *pd;
    uint32_t handle;
};

/* Stub functions - return error or NULL */
static inline struct ibv_device **ibv_get_device_list(int *n) { if(n) *n = 0; return NULL; }
static inline void ibv_free_device_list(struct ibv_device **list) { (void)list; }
static inline const char *ibv_get_device_name(struct ibv_device *dev) { (void)dev; return "stub"; }
static inline struct ibv_context *ibv_open_device(struct ibv_device *dev) { (void)dev; return NULL; }
static inline int ibv_close_device(struct ibv_context *ctx) { (void)ctx; return 0; }
static inline struct ibv_pd *ibv_alloc_pd(struct ibv_context *ctx) { (void)ctx; return NULL; }
static inline int ibv_dealloc_pd(struct ibv_pd *pd) { (void)pd; return 0; }
static inline struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t len, int flags) {
    (void)pd; (void)addr; (void)len; (void)flags; return NULL;
}
static inline int ibv_dereg_mr(struct ibv_mr *mr) { (void)mr; return 0; }
static inline struct ibv_cq *ibv_create_cq(struct ibv_context *ctx, int cqe, void *ctx2, void *ch, int vec) {
    (void)ctx; (void)cqe; (void)ctx2; (void)ch; (void)vec; return NULL;
}
static inline int ibv_destroy_cq(struct ibv_cq *cq) { (void)cq; return 0; }
static inline int ibv_poll_cq(struct ibv_cq *cq, int n, struct ibv_wc *wc) { (void)cq; (void)n; (void)wc; return 0; }
static inline struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *attr) {
    (void)pd; (void)attr; return NULL;
}
static inline int ibv_destroy_qp(struct ibv_qp *qp) { (void)qp; return 0; }
static inline int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int mask) {
    (void)qp; (void)attr; (void)mask; return -1;
}
static inline int ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int mask, struct ibv_qp_init_attr *init_attr) {
    (void)qp; (void)attr; (void)mask; (void)init_attr; return -1;
}
static inline int ibv_query_gid(struct ibv_context *ctx, int port, int idx, union ibv_gid *gid) {
    (void)ctx; (void)port; (void)idx; (void)gid; return -1;
}
static inline int ibv_query_port(struct ibv_context *ctx, int port, struct ibv_port_attr *attr) {
    (void)ctx; (void)port; (void)attr; return -1;
}
static inline int ibv_query_device(struct ibv_context *ctx, struct ibv_device_attr *attr) {
    (void)ctx; (void)attr; return -1;
}
static inline int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad) {
    (void)qp; (void)wr; (void)bad; return -1;
}
static inline int ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad) {
    (void)qp; (void)wr; (void)bad; return -1;
}
static inline const char *ibv_wc_status_str(enum ibv_wc_status status) {
    (void)status; return "stub_error";
}

/* NCCL-001: Async event monitoring stubs */
static inline int ibv_get_async_event(struct ibv_context *ctx, struct ibv_async_event *event) {
    (void)ctx; (void)event; return -1;
}
static inline void ibv_ack_async_event(struct ibv_async_event *event) { (void)event; }

/* NCCL-001: Event type to string helper */
static inline const char *ibv_event_type_str(enum ibv_event_type event_type) {
    switch (event_type) {
        case IBV_EVENT_CQ_ERR: return "IBV_EVENT_CQ_ERR";
        case IBV_EVENT_QP_FATAL: return "IBV_EVENT_QP_FATAL";
        case IBV_EVENT_QP_REQ_ERR: return "IBV_EVENT_QP_REQ_ERR";
        case IBV_EVENT_QP_ACCESS_ERR: return "IBV_EVENT_QP_ACCESS_ERR";
        case IBV_EVENT_COMM_EST: return "IBV_EVENT_COMM_EST";
        case IBV_EVENT_SQ_DRAINED: return "IBV_EVENT_SQ_DRAINED";
        case IBV_EVENT_PATH_MIG: return "IBV_EVENT_PATH_MIG";
        case IBV_EVENT_PATH_MIG_ERR: return "IBV_EVENT_PATH_MIG_ERR";
        case IBV_EVENT_DEVICE_FATAL: return "IBV_EVENT_DEVICE_FATAL";
        case IBV_EVENT_PORT_ACTIVE: return "IBV_EVENT_PORT_ACTIVE";
        case IBV_EVENT_PORT_ERR: return "IBV_EVENT_PORT_ERR";
        case IBV_EVENT_LID_CHANGE: return "IBV_EVENT_LID_CHANGE";
        case IBV_EVENT_PKEY_CHANGE: return "IBV_EVENT_PKEY_CHANGE";
        case IBV_EVENT_SM_CHANGE: return "IBV_EVENT_SM_CHANGE";
        case IBV_EVENT_SRQ_ERR: return "IBV_EVENT_SRQ_ERR";
        case IBV_EVENT_SRQ_LIMIT_REACHED: return "IBV_EVENT_SRQ_LIMIT_REACHED";
        case IBV_EVENT_QP_LAST_WQE_REACHED: return "IBV_EVENT_QP_LAST_WQE_REACHED";
        case IBV_EVENT_CLIENT_REREGISTER: return "IBV_EVENT_CLIENT_REREGISTER";
        case IBV_EVENT_GID_CHANGE: return "IBV_EVENT_GID_CHANGE";
        default: return "UNKNOWN_EVENT";
    }
}

#endif /* INFINIBAND_VERBS_H */

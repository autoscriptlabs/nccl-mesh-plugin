# NCCL Mesh Plugin Architecture

This document provides a deep dive into the architecture and implementation of the NCCL Mesh Plugin.

## Overview

The NCCL Mesh Plugin is a custom network transport that enables NCCL to work with direct-connect RDMA mesh topologies where each node pair is on a different subnet. This is a configuration that standard NCCL plugins cannot handle.

## The Problem

### Standard NCCL Networking

NCCL's built-in network plugins assume one of two scenarios:

1. **InfiniBand Fabric**: All nodes connected through IB switches, sharing a single subnet
2. **TCP/IP Sockets**: Standard IP networking with routing

### Our Topology

```
     Node A (192.168.100.2, 192.168.101.2)
              /                \
    192.168.100.x         192.168.101.x
            /                    \
    Node C                      Node B
(192.168.100.3,            (192.168.101.3,
 192.168.102.3)             192.168.102.2)
            \                    /
             \   192.168.102.x  /
              \                /
               \--------------/
```

Each link is on a **different subnet**:
- A↔B: 192.168.101.0/24
- A↔C: 192.168.100.0/24
- B↔C: 192.168.102.0/24

This means:
- No single IP can reach all peers
- Standard IB plugin fails (expects single subnet)
- TCP socket plugin would need IP routing (adds latency)

## Solution Architecture

### Key Insight

Each node has **multiple NICs**, each on a different subnet. When connecting to a peer, we must:
1. Determine which subnet the peer is on
2. Use the local NIC on that same subnet
3. Establish RDMA connection over that specific NIC pair

### Handle Structure

The NCCL handle is expanded to advertise **all** local addresses:

```c
struct mesh_handle {
    uint32_t magic;              // Validation
    uint8_t  num_addrs;          // Number of addresses
    uint16_t handshake_port;     // TCP port for QP exchange
    
    struct mesh_addr_entry {
        uint32_t ip;             // IP address (network order)
        uint32_t mask;           // Subnet mask
        uint32_t qp_num;         // Queue Pair number
        uint8_t  nic_idx;        // Index into local NIC array
    } addrs[MESH_MAX_ADDRS];
};
```

### Connection Flow

#### Phase 1: Listen

```c
ncclResult_t mesh_listen(int dev, void *handle, void **listenComm) {
    // 1. Create QPs on ALL local NICs
    for (int i = 0; i < num_nics; i++) {
        create_qp_on_nic(&nics[i]);
    }
    
    // 2. Start background handshake thread
    pthread_create(&thread, handshake_thread_func, lcomm);
    
    // 3. Fill handle with ALL addresses
    for (int i = 0; i < num_nics; i++) {
        handle->addrs[i].ip = nics[i].ip_addr;
        handle->addrs[i].mask = nics[i].netmask;
        handle->addrs[i].qp_num = qps[i]->qp_num;
    }
}
```

#### Phase 2: Connect

```c
ncclResult_t mesh_connect(int dev, void *handle, void **sendComm) {
    // 1. Search peer's addresses for reachable one
    for (int i = 0; i < handle->num_addrs; i++) {
        uint32_t peer_subnet = handle->addrs[i].ip & handle->addrs[i].mask;
        
        // Find local NIC on same subnet
        for (int j = 0; j < num_local_nics; j++) {
            if (local_nics[j].subnet == peer_subnet) {
                selected_nic = &local_nics[j];
                selected_peer_addr = &handle->addrs[i];
                break;
            }
        }
    }
    
    // 2. Create QP on selected NIC
    create_qp_on_nic(selected_nic);
    
    // 3. Exchange QP info via TCP handshake
    send_handshake(peer_ip, peer_port, &local_qp_info, &remote_qp_info);
    
    // 4. Connect QP to peer's QP
    connect_qp(local_qp, remote_qp_info);
}
```

#### Phase 3: Accept

```c
ncclResult_t mesh_accept(void *listenComm, void **recvComm) {
    // Get pre-connected QP from handshake thread's queue
    pthread_mutex_lock(&queue_mutex);
    while (queue_empty) {
        pthread_cond_wait(&queue_cond, &queue_mutex);
    }
    entry = dequeue();
    pthread_mutex_unlock(&queue_mutex);
    
    // Return the ready connection
    rcomm->qp = entry->local_qp;
    rcomm->nic = entry->nic;
}
```

### Background Handshake Thread

The handshake thread solves a critical deadlock problem:

**Without thread:**
```
Rank 0: connect() → TCP connect to Rank 1 → blocks waiting for accept()
Rank 1: connect() → TCP connect to Rank 0 → blocks waiting for accept()
// DEADLOCK: Neither can call accept() because both stuck in connect()
```

**With thread:**
```
Rank 0: listen() starts thread → thread waits for TCP connections
Rank 1: listen() starts thread → thread waits for TCP connections
Rank 0: connect() → TCP connects to Rank 1's thread → gets response → returns
Rank 1: connect() → TCP connects to Rank 0's thread → gets response → returns
Rank 0: accept() → gets QP from queue (filled by thread) → returns
Rank 1: accept() → gets QP from queue (filled by thread) → returns
// SUCCESS: Thread handles incoming connections asynchronously
```

### RDMA Queue Pair Setup

Each connection requires proper QP state transitions:

```
RESET → INIT → RTR → RTS
```

```c
int mesh_connect_qp(struct ibv_qp *qp, struct mesh_nic *nic,
                    struct mesh_handle *remote) {
    // RESET → INIT
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = nic->port_num;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | 
                              IBV_ACCESS_REMOTE_READ |
                              IBV_ACCESS_LOCAL_WRITE;
    ibv_modify_qp(qp, &qp_attr, ...);
    
    // INIT → RTR (Ready to Receive)
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_4096;
    qp_attr.dest_qp_num = remote->qp_num;
    qp_attr.rq_psn = remote->psn;
    qp_attr.ah_attr.dlid = remote->lid;  // 0 for RoCE
    qp_attr.ah_attr.grh.dgid = remote->gid;  // Peer's GID
    ibv_modify_qp(qp, &qp_attr, ...);
    
    // RTR → RTS (Ready to Send)
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = local_psn;
    qp_attr.timeout = 14;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    ibv_modify_qp(qp, &qp_attr, ...);
}
```

### Data Transfer

#### Send Path

```c
ncclResult_t mesh_isend(void *sendComm, void *data, int size,
                        void *mhandle, void **request) {
    struct ibv_send_wr wr = {
        .wr_id = (uint64_t)req,
        .sg_list = &sge,
        .num_sge = 1,
        .opcode = IBV_WR_SEND,
        .send_flags = IBV_SEND_SIGNALED,
    };
    
    sge.addr = (uint64_t)data;
    sge.length = size;
    sge.lkey = mr->lkey;
    
    ibv_post_send(comm->qp, &wr, &bad_wr);
}
```

#### Receive Path

```c
ncclResult_t mesh_irecv(void *recvComm, int n, void **data,
                        int *sizes, void **mhandles, void **request) {
    struct ibv_recv_wr wr = {
        .wr_id = (uint64_t)req,
        .sg_list = &sge,
        .num_sge = 1,
    };
    
    sge.addr = (uint64_t)data[0];
    sge.length = sizes[0];
    sge.lkey = mr->lkey;
    
    ibv_post_recv(comm->qp, &wr, &bad_wr);
}
```

#### Completion Polling

```c
ncclResult_t mesh_test(void *request, int *done, int *sizes) {
    struct ibv_wc wc;
    
    int ret = ibv_poll_cq(req->cq, 1, &wc);
    if (ret > 0) {
        if (wc.status == IBV_WC_SUCCESS) {
            *done = 1;
            if (sizes) *sizes = wc.byte_len;
        } else {
            // Handle error
        }
    } else {
        *done = 0;  // Not complete yet
    }
}
```

## Memory Registration

RDMA requires memory to be registered with the NIC:

```c
ncclResult_t mesh_regMr(void *comm, void *data, size_t size,
                        int type, void **mhandle) {
    int access = IBV_ACCESS_LOCAL_WRITE | 
                 IBV_ACCESS_REMOTE_WRITE |
                 IBV_ACCESS_REMOTE_READ;
    
    mrh->mr = ibv_reg_mr(nic->pd, data, size, access);
    *mhandle = mrh;
}
```

**Unified Memory Note**: On Grace Hopper / DGX Spark systems, GPU and CPU share the same physical memory via NVLink-C2C. This means RDMA registration works directly on GPU-accessible memory without any staging copies - we get GPUDirect-like semantics automatically.

## Performance Considerations

### Achieved Performance

- **8+ GB/s** effective bandwidth
- **~64%** of 100 Gbps line rate
- Zero-copy on unified memory (Grace Hopper)
- Sufficient for distributed ML workloads

### Current Bottlenecks

1. **Single QP**: One Queue Pair per connection limits parallelism
2. **Completion Signaling**: Every operation signals completion
3. **Protocol Overhead**: RC transport has per-message overhead

### Future Optimizations

1. **Multi-QP**: Multiple QPs per connection for parallelism
2. **Selective Signaling**: Signal every N operations
3. **Inline Data**: Small messages embedded in WQE

## File Structure

```
nccl-mesh-plugin/
├── src/
│   └── mesh_plugin.c      # Main implementation (~1400 lines)
├── include/
│   └── mesh_plugin.h      # Data structures and declarations
├── nccl/
│   ├── net.h              # NCCL net plugin interface
│   ├── net_v8.h           # v8 properties structure
│   └── err.h              # NCCL error codes
└── Makefile
```

## Debugging

Enable debug output:

```bash
export NCCL_DEBUG=INFO
export NCCL_MESH_DEBUG=1
```

Common issues:

1. **"No local NIC found"**: Subnet mismatch, check IP configuration
2. **"Handshake timeout"**: Firewall blocking TCP, check ports
3. **"QP transition failed"**: GID index wrong, try different `NCCL_MESH_GID_INDEX`
4. **"WC error status=12"**: Transport retry exceeded, check RDMA connectivity

## Conclusion

The NCCL Mesh Plugin demonstrates that with careful engineering, NCCL can be extended to support unconventional network topologies. The key innovations—multi-address handles, subnet-aware NIC selection, and asynchronous handshaking—provide a template for other custom NCCL transports.

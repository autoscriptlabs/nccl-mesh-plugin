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

**Note on Memory Architecture**: On Grace Hopper / DGX Spark systems with unified memory, GPU and CPU share the same physical memory pool. Memory registered for RDMA is directly accessible by the GPU with no copy overhead. On discrete GPU systems, host memory staging would apply (GPU→Host→RDMA→Host→GPU).

## Topology Routing Layer

The routing layer (`mesh_routing.c`) enables communication in partial mesh topologies where not all nodes are directly connected.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NCCL Application                          │
├─────────────────────────────────────────────────────────────┤
│                   NCCL Framework                             │
├─────────────────────────────────────────────────────────────┤
│                 NCCL Net Plugin API                          │
│  (listen, connect, accept, send, recv, test, close)         │
├─────────────────────────────────────────────────────────────┤
│                  Mesh Plugin Router                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Topology Discovery │ Routing Table │ Path Selection │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│               Mesh Plugin Core                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Subnet-Aware NIC │ QP Management │ Connection Pool  │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    RDMA Verbs API                            │
└─────────────────────────────────────────────────────────────┘
```

### NIC Lane Classification

NICs are classified by link speed into two lanes:
- **Fast Lane** (≥50 Gbps): Used for collective traffic (all-reduce, etc.)
- **Management Lane** (<50 Gbps): Used for orchestration, checkpoints

```c
enum mesh_nic_lane {
    MESH_LANE_UNKNOWN = 0,
    MESH_LANE_MANAGEMENT,  // 10GbE, 25GbE
    MESH_LANE_FAST,        // 100GbE, 200GbE
};
```

This separation allows 10GbE management networks (all-to-all via switch) to coexist with fast direct-connect links in ring/line topologies.

### Topology Detection

The plugin automatically detects topology based on node degree analysis:

```c
enum mesh_topology_type {
    MESH_TOPO_FULL_MESH,  // All nodes directly connected
    MESH_TOPO_RING,       // Each node has exactly 2 neighbors
    MESH_TOPO_LINE,       // 2 endpoints (1 neighbor), rest have 2
    MESH_TOPO_PARTIAL,    // Mixed connectivity
};
```

### Routing Table

BFS shortest-path algorithm computes routes to all nodes:

```c
struct mesh_route_entry {
    uint32_t dest_node_id;      // Destination
    uint8_t  num_hops;          // 1 = direct, 2+ = relayed
    int      is_direct;         // Direct connection available?
    uint32_t next_hop_node_id;  // First hop for relay
    uint8_t  relay_path[];      // Full path for multi-hop
};
```

### Ring Topology Optimizations

For ring topologies, dual-path routing enables load balancing:

```
Ring: A - B - C - D - A

A to C: Two paths available
  - Clockwise:  A → B → C (2 hops)
  - Counter-CW: A → D → C (2 hops)
```

The plugin tracks bytes sent on each path and balances load:
- `NCCL_MESH_RING_LOAD_BALANCE=1`: Enable load balancing (default)
- `NCCL_MESH_RING_BALANCE_THRESHOLD`: Bytes difference before switching

### Line Topology Optimizations

For line topologies, endpoints and direction are detected:

```
Line: A - B - C - D

Endpoints: A (head), D (tail)
Direction from B to D: towards tail (+1)
Direction from B to A: towards head (-1)
```

### Relay Communication

Non-adjacent nodes communicate via store-and-forward relay:

```c
struct mesh_relay_header {
    uint32_t magic;           // MESH_RELAY_MAGIC
    uint32_t src_node_id;     // Original sender
    uint32_t dst_node_id;     // Final destination
    uint32_t payload_size;    // Data size
    uint8_t  hop_count;       // Current hop
    uint8_t  path[];          // Full path
};
```

Relay nodes receive data, check the header, and forward to the next hop.

## Performance Considerations

### Current Bottlenecks

1. **Single QP**: One Queue Pair per connection limits parallelism
2. **Completion Signaling**: Every operation signals completion
3. **Store-and-Forward Relay**: Adds ~1 RTT latency per hop for non-adjacent nodes

### Achieved Performance

- **200 Gbps** link speed per QSFP56 cable (ConnectX-7, dual PCIe 5.0 x4 channels)
- **8+ GB/s** effective bandwidth measured on earlier 100Gbps config (~64% utilization)
- Working relay routing for non-adjacent nodes in ring/line topologies
- Sufficient for distributed ML training (DeepSpeed ZeRO-3) and inference (vLLM)

### Implemented Optimizations

#### Ring/Line Topology Support
4+ node clusters without full mesh connectivity:
- Relay routing through intermediate nodes
- Store-and-forward with 4MB relay buffers
- Automatic topology discovery and BFS path selection
- Dual-path load balancing for ring topologies

### Planned Improvements

#### Cut-Through Forwarding
Reduce relay latency by forwarding packets as they arrive:
- Don't wait for complete message before forwarding
- Pipeline-friendly for large transfers

#### Additional Optimizations
1. **Multi-QP**: Multiple QPs per connection for parallelism
2. **Selective Signaling**: Signal every N operations to reduce CQ overhead
3. **Inline Data**: Small messages embedded in WQE

## File Structure

```
nccl-mesh-plugin/
├── src/
│   ├── mesh_plugin.c      # Main plugin implementation (~4400 lines)
│   └── mesh_routing.c     # Routing and relay layer (~3100 lines)
├── include/
│   ├── mesh_plugin.h      # Plugin data structures
│   └── mesh_routing.h     # Routing data structures
├── tests/
│   ├── test_routing.c     # Unit tests for routing (13 tests)
│   ├── test_ring_topo.py  # Ring topology integration tests
│   └── test_line_topo.py  # Line topology integration tests
├── patches/
│   └── ticket-f-vllm-nccl-init-barrier.patch  # vLLM upstream fix
├── nccl/
│   ├── net.h              # NCCL net plugin interface
│   ├── net_v8.h           # v8 properties structure
│   └── err.h              # NCCL error codes
├── docs/
│   ├── ARCHITECTURE.md    # This file
│   ├── SETUP.md           # Hardware setup guide
│   └── PARTIAL_MESH_ROUTING_PLAN.md  # Routing implementation plan
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

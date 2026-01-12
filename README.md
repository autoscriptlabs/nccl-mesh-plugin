# NCCL Mesh Plugin

**Custom NCCL network plugin enabling distributed ML over direct-connect RDMA mesh topologies.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What This Does

This plugin enables NCCL (NVIDIA Collective Communications Library) to work with **direct-connect mesh topologies** where each node pair is on a different subnet. Standard NCCL plugins assume either:
- A switched InfiniBand fabric (all nodes on same subnet)
- TCP/IP networking (slow, high latency)

Neither works for direct-cabled RDMA meshes. This plugin does.

## 🔧 The Problem We Solved

```
                    ┌─────────────┐
                    │   Spark-A   │
                    │  (titanic)  │
                    └──────┬──────┘
           192.168.101.x   │   192.168.100.x
              (100Gbps)    │      (100Gbps)
                    ┌──────┴──────┐
                    │             │
              ┌─────┴─────┐ ┌─────┴─────┐
              │  Spark-B  │ │  Spark-C  │
              │ (iceberg) │ │(carpathia)│
              └─────┬─────┘ └─────┬─────┘
                    │             │
                    └──────┬──────┘
                    192.168.102.x
                      (100Gbps)
```

**Three DGX Spark workstations** connected in a triangle mesh with direct 100Gbps RDMA cables. Each link is on a **different subnet** - a configuration NVIDIA never intended to support.

## 🚀 Results

| Metric | Value |
|--------|-------|
| Effective Bandwidth | **8+ GB/s** |
| Line Rate Utilization | ~64% |
| Topology | 3-node triangle mesh |
| Link Speed | 100 Gbps per link |

Successfully ran **distributed LLM inference** (Mistral-7B) across all 3 nodes using NCCL over this custom topology.

## 🏗️ Architecture

### Key Innovations

1. **Multi-Address Handle Exchange**
   - Each node advertises ALL its subnet IPs in the NCCL handle
   - Connector searches for reachable addresses by subnet matching

2. **Subnet-Aware NIC Selection**
   - `connect()` finds the local NIC on the same subnet as the peer
   - Automatic routing without IP forwarding or bridges

3. **Background Handshake Thread**
   - Eliminates deadlock when both ranks call `connect()` simultaneously
   - TCP-based QP info exchange runs asynchronously

4. **Bidirectional QP Exchange**
   - Each connection creates fresh Queue Pairs on both sides
   - No QP reuse across multiple NCCL channels

### RDMA Implementation

- Raw InfiniBand Verbs API (libibverbs)
- Reliable Connected (RC) Queue Pairs
- RoCE v2 over Ethernet

> **Note on Grace Hopper / DGX Spark**: These systems use unified memory where GPU and CPU share the same physical memory pool. There is no GPU↔Host copy overhead—memory registered for RDMA is directly accessible by the GPU.

## 📦 Installation

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install libibverbs-dev librdmacm-dev

# Verify RDMA devices
ibv_devices
```

### Build

```bash
git clone https://github.com/yourusername/nccl-mesh-plugin.git
cd nccl-mesh-plugin
make
```

### Use

```bash
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export NCCL_NET_PLUGIN=mesh
export NCCL_DEBUG=INFO  # or WARN for less output

# Run your distributed job
python your_distributed_script.py
```

## 🧪 Testing

### Basic All-Reduce Test

```python
import torch
import torch.distributed as dist

dist.init_process_group('nccl', rank=RANK, world_size=3,
    init_method='tcp://MASTER_IP:29500')

t = torch.ones(1000, device='cuda')
dist.all_reduce(t)
print(f'Result: {t[0]}')  # Should print 3.0

dist.destroy_process_group()
```

### Bandwidth Benchmark

```python
import torch
import torch.distributed as dist
import time

dist.init_process_group('nccl', rank=RANK, world_size=3,
    init_method='tcp://MASTER_IP:29500')

t = torch.ones(1024*1024*64, device='cuda')  # 256MB

# Warmup
for _ in range(5):
    dist.all_reduce(t)
torch.cuda.synchronize()

# Benchmark
start = time.time()
for _ in range(20):
    dist.all_reduce(t)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f'Bandwidth: {(256*20/1024)/elapsed:.2f} GB/s')
```

## 🔬 How It Works

### Connection Flow

```
Rank 0 (listen)                    Rank 1 (connect)
     │                                   │
     ▼                                   │
 listen()                                │
 ├─ Create QPs on ALL NICs               │
 ├─ Start handshake thread               │
 ├─ Return handle with all IPs           │
     │                                   │
     │◄──────── handle exchange ────────►│
     │                                   │
     │                                   ▼
     │                              connect()
     │                              ├─ Find matching subnet
     │                              ├─ Create QP on that NIC
     │                              ├─ TCP handshake ──────────►│
     │                                   │                      │
     │◄────────────────────────────────────────── QP info ─────┤
     │                                   │                      │
     ▼                                   ▼                      ▼
 accept()                           Connect QP            [handshake thread]
 ├─ Get QP from queue               to peer's QP          ├─ Accept TCP
 └─ Return recv_comm                     │                ├─ Create new QP
                                         │                ├─ Connect QPs
                                         │                └─ Queue for accept()
                                         │
                                    ┌────┴────┐
                                    │ RDMA OK │
                                    └─────────┘
```

### Subnet Matching

```c
// For each peer address in handle
for (int i = 0; i < handle->num_addrs; i++) {
    uint32_t peer_ip = handle->addrs[i].ip;
    
    // Find local NIC on same subnet
    for (int j = 0; j < num_nics; j++) {
        if ((peer_ip & nic[j].netmask) == nic[j].subnet) {
            // Found matching NIC!
            selected_nic = &nic[j];
            break;
        }
    }
}
```

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `NCCL_NET_PLUGIN` | - | Set to `mesh` to use this plugin |
| `NCCL_DEBUG` | `WARN` | Set to `INFO` for detailed logs |
| `NCCL_MESH_GID_INDEX` | `3` | RoCE GID index to use |
| `NCCL_MESH_DEBUG` | `0` | Enable plugin debug output |

## 🚧 Current Limitations

- **Full mesh required**: Non-adjacent nodes can't communicate yet (no relay routing)
- **Single channel per port**: Currently uses one PCIe lane per ConnectX-7 port (100Gbps), not both (200Gbps)
- **Single QP per connection**: No multi-rail aggregation
- **RoCE v2 only**: No InfiniBand support (Ethernet only)

## 🗺️ Roadmap

### Near-term
- [ ] **Ring topology support**: Relay routing for non-adjacent nodes (enables 4+ node clusters without full mesh)
- [ ] **Dual-channel per port**: Utilize both PCIe 5.0 x4 lanes per ConnectX-7 port for 200Gbps per cable
- [ ] **Robustness improvements**: Better error handling, recovery, and diagnostics

### Future
- [ ] Multi-QP per connection for higher bandwidth
- [ ] Adaptive routing for partial meshes
- [ ] Performance tuning (inline data, selective signaling)

## 📚 References

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [RDMA Aware Networks Programming User Manual](https://www.mellanox.com/related-docs/prod_software/RDMA_Aware_Programming_user_manual.pdf)
- [InfiniBand Verbs API](https://github.com/linux-rdma/rdma-core)

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

Built to connect three DGX Spark workstations that NVIDIA never intended to be clustered. Sometimes the best solutions come from ignoring "supported configurations."

---

*"The future of distributed AI computing is here."* - Mistral-7B, running on this very plugin

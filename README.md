# NCCL Mesh Plugin

**Custom NCCL network plugin enabling distributed ML over direct-connect RDMA mesh topologies.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This plugin enables NCCL (NVIDIA Collective Communications Library) to work with **direct-connect mesh topologies** where each node pair is on a different subnet. Standard NCCL plugins assume either a switched InfiniBand fabric (all nodes on same subnet) or TCP/IP networking (slow, high latency). Neither works for direct-cabled RDMA meshes. This plugin does.

**Supported topologies:**
- **Full mesh** (3 nodes): Every node directly connected to every other
- **Ring** (4+ nodes): Each node connects to 2 neighbors, relay routing for non-adjacent
- **Line** (any number): Chain of nodes, relay routing for multi-hop communication

**Current production configuration**: 4x DGX Spark workstations in a ring topology with **200Gbps** QSFP56 direct RDMA links (ConnectX-7, dual-channel), running distributed LLM training (Qwen2.5-14B) with DeepSpeed ZeRO-3. Also supports vLLM inference (with [upstream patches](#vllm-support)).

## Quick Start

```bash
# 1. Build the plugin
git clone https://github.com/yourusername/nccl-mesh-plugin.git
cd nccl-mesh-plugin
make

# 2. Set environment variables
export NCCL_NET_PLUGIN=$(pwd)/libnccl-net.so
export NCCL_SOCKET_IFNAME=eth0  # Your management network interface

# 3. Run distributed training (with SLURM)
salloc -N4 --exclusive ./examples/run_qwen14b_deepspeed.sh --steps 100
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

## The Problem We Solved

```
        Node A (titanic)
       /      \
   200Gbps    200Gbps
   QSFP56     QSFP56
     /          \
  Node B        Node D
(iceberg)    (carpathia-2)
     \          /
   200Gbps    200Gbps
   QSFP56     QSFP56
     \          /
      \        /
       Node C
    (carpathia)
```

**Four DGX Spark workstations** in a ring topology with direct **200Gbps** QSFP56 RDMA cables (ConnectX-7, dual PCIe 5.0 x4 channels). Each link is on a **different subnet** — a configuration not covered by standard NCCL network plugins. Non-adjacent nodes (A↔C, B↔D) communicate via automatic relay routing through their neighbors.

## Performance

| Metric | Value |
|--------|-------|
| Effective Bandwidth | **8+ GB/s** (direct), relay adds ~1 RTT/hop |
| Line Rate Utilization | ~64% of single channel* |
| Topology | 4-node ring (production) |
| Link Speed | **200 Gbps** per link (QSFP56, dual-channel) |
| Peak AllReduce | 16.93 GB/s (1000 MB messages) |

*Benchmarks measured on earlier 100Gbps single-channel config. 200Gbps dual-channel numbers pending.

## LLM Training Example

This plugin was developed for distributed LLM training. Here's how to train Qwen2.5-14B across 4 nodes:

### Prerequisites

- PyTorch 2.0+
- DeepSpeed
- Transformers
- FlashAttention 2 (optional, for memory efficiency)

### Running Training

**Interactive (for testing):**
```bash
salloc -N4 --exclusive
./examples/run_qwen14b_deepspeed.sh --steps 100
```

**Batch job (for full training):**
```bash
# Submit with default settings (12000 steps, ~1 epoch)
sbatch examples/submit_training.sbatch

# Or customize: steps, warmup_steps, learning_rate
sbatch examples/submit_training.sbatch 12000 100 2e-5

# Monitor
tail -f training_<jobid>.log
```

### Memory Usage

With DeepSpeed ZeRO-3 on 4 nodes (117GB unified memory each):

| Component | Per-Node Memory |
|-----------|-----------------|
| Sharded parameters | ~9 GB |
| Sharded optimizer states | ~19 GB |
| Sharded gradients | ~9 GB |
| **Total allocated** | **~38 GB** |
| Reserved (PyTorch cache) | ~42 GB |

Plenty of headroom for larger batch sizes or longer sequences.

### Scaling

| Model Size | Minimum Nodes | Recommended |
|------------|---------------|-------------|
| 7-14B | 2 | 3 |
| 32B | 3 | 4 |
| 70B | 6 | 8 |

## Installation

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

### Environment Setup

```bash
# Required
export NCCL_NET_PLUGIN=/path/to/libnccl-net.so
export LD_LIBRARY_PATH=/path/to/nccl-mesh-plugin:$LD_LIBRARY_PATH
export NCCL_SOCKET_IFNAME=eth0  # Bootstrap interface (management network)

# Optional
export NCCL_DEBUG=INFO          # Or WARN for less output
export NCCL_MESH_GID_INDEX=3    # RoCE GID index (try 0-3 if issues)
```

## Architecture

### Key Innovations

1. **Multi-Address Handle Exchange**: Each node advertises ALL its subnet IPs in the NCCL handle
2. **Subnet-Aware NIC Selection**: `connect()` finds the local NIC on the same subnet as the peer
3. **Background Handshake Thread**: Eliminates deadlock when both ranks call `connect()` simultaneously
4. **Bidirectional QP Exchange**: Fresh Queue Pairs created for each connection

### Connection Flow

```
Rank 0 (listen)                    Rank 1 (connect)
     |                                   |
     v                                   |
 listen()                                |
 +- Create QPs on ALL NICs               |
 +- Start handshake thread               |
 +- Return handle with all IPs           |
     |                                   |
     |<-------- handle exchange -------->|
     |                                   |
     |                                   v
     |                              connect()
     |                              +- Find matching subnet
     |                              +- Create QP on that NIC
     |                              +- TCP handshake ---------->|
     |                                   |                      |
     |<------------------------------ QP info -----------------|
     |                                   |                      |
     v                                   v                      v
 accept()                           Connect QP            [handshake thread]
 +- Get QP from queue               to peer's QP          +- Accept TCP
 +- Return recv_comm                     |                +- Create new QP
                                         |                +- Connect QPs
                                         |                +- Queue for accept()
                                    +----+----+
                                    | RDMA OK |
                                    +---------+
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for implementation details.

## Configuration Reference

### Core Settings

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `NCCL_NET_PLUGIN` | - | Path to `libnccl-net.so` |
| `NCCL_SOCKET_IFNAME` | - | Bootstrap network interface |
| `NCCL_DEBUG` | `WARN` | Log level: `INFO`, `WARN`, `TRACE` |
| `NCCL_MESH_GID_INDEX` | `3` | RoCE GID index to use |
| `NCCL_MESH_DEBUG` | `0` | Plugin debug: 0=off, 1=info, 2=verbose |
| `NCCL_MESH_TIMEOUT_MS` | `5000` | Connection timeout (ms) |
| `NCCL_MESH_RETRY_COUNT` | `3` | Connection retry attempts |

### Topology & Routing Settings

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `NCCL_MESH_ENABLE_RELAY` | `1` | Enable relay routing for non-adjacent nodes |
| `NCCL_MESH_MAX_HOPS` | `4` | Maximum relay hops allowed |
| `NCCL_MESH_RING_LOAD_BALANCE` | `1` | Enable load balancing across ring paths |
| `NCCL_MESH_RING_PREFER_SHORT` | `0` | Always prefer shorter path (disable load balance) |
| `NCCL_MESH_RING_BALANCE_THRESHOLD` | `1048576` | Bytes difference before switching paths (1MB) |

## Project Structure

```
nccl-mesh-plugin/
+-- src/
|   +-- mesh_plugin.c          # Main plugin implementation
|   +-- mesh_routing.c         # Topology routing and relay layer
+-- include/
|   +-- mesh_plugin.h          # Plugin data structures
|   +-- mesh_routing.h         # Routing data structures
+-- nccl/                      # NCCL header files (net.h, net_v8.h)
+-- tests/
|   +-- test_routing.c         # Unit tests for routing
|   +-- test_ring_topo.py      # Ring topology integration tests
|   +-- test_line_topo.py      # Line topology integration tests
+-- examples/
|   +-- train_qwen14b_deepspeed.py   # LLM training script
|   +-- run_qwen14b_deepspeed.sh     # Launcher script
|   +-- submit_training.sbatch       # SLURM batch submission
|   +-- test_allreduce.py            # Basic communication test
|   +-- benchmark_bandwidth.py       # Bandwidth benchmark
+-- patches/
|   +-- ticket-f-vllm-nccl-init-barrier.patch  # vLLM upstream fix
+-- docs/
|   +-- ARCHITECTURE.md              # Deep dive into implementation
|   +-- SETUP.md                     # Hardware setup guide
|   +-- PARTIAL_MESH_ROUTING_PLAN.md # Routing implementation plan
+-- QUICKSTART.md              # Step-by-step getting started
+-- BENCHMARKS.md              # Performance measurements
+-- Makefile
```

## Troubleshooting

### Plugin not loading

```
NET/Plugin: Could not find: /path/to/libnccl-net.so
```

- Verify the path is correct on ALL nodes
- Check file permissions
- Rebuild with `make clean && make`

### Connection timeout with 192.168.x.x addresses

```
Call to ibv_modify_qp failed with 110 Connection timed out
local GID ::ffff:192.168.100.1, remote GID ::ffff:192.168.101.3
```

The mesh plugin didn't load, and NCCL fell back to the built-in IB plugin which uses unroutable addresses. Fix:
- Ensure `NCCL_NET_PLUGIN` points to the correct path
- Look for `Loaded net plugin Mesh (v9)` in logs

### GID index issues

Try different values: `export NCCL_MESH_GID_INDEX=0` (or 1, 2, 3)

Check available GIDs:
```bash
show_gids
# or
cat /sys/class/infiniband/*/ports/1/gids/*
```

## Limitations

- **RoCE v2 only**: No InfiniBand fabric support
- **Store-and-forward relay**: Adds ~1 RTT latency per hop for non-adjacent nodes (cut-through planned)
- **vLLM requires upstream patches**: See [vLLM Support](#vllm-support) below

## vLLM Support

The plugin works with vLLM for distributed inference, but requires a small upstream patch to fix a race condition in NCCL communicator initialization. The patch adds a barrier + configurable delay between `torch.distributed.init_process_group()` and NCCL communicator creation, giving the mesh plugin time to complete RDMA setup.

Apply the patch (included in `patches/`):
```bash
cd /path/to/vllm
git apply /path/to/nccl-mesh-plugin/patches/ticket-f-vllm-nccl-init-barrier.patch

# Then run vLLM with the delay
export VLLM_NCCL_INIT_DELAY=2.0  # seconds
```

This patch has been submitted upstream but is not yet merged. See the patch file for details on the two modified files (`vllm/distributed/parallel_state.py` and `vllm/envs.py`).

## Roadmap

- [x] Ring topology support (for 4+ nodes with only 2 NICs each)
- [x] Line topology support (chain of nodes)
- [x] Relay routing for non-adjacent nodes
- [x] Automatic topology detection
- [x] Dual-path routing with load balancing (ring)
- [x] Dual-channel per port (200Gbps QSFP56)
- [x] vLLM support (with upstream patch)
- [x] ARM64/AArch64 memory ordering fixes
- [x] Blackwell PCIe topology / NIC classification support
- [x] Automatic GID index discovery (4+ nodes)
- [x] Server metrics logging for transport stall diagnosis
- [x] Hardened error paths (eliminate silent hangs)
- [ ] Cut-through forwarding (reduce relay latency)
- [ ] Multi-QP aggregation

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - Get running in 15 minutes
- [docs/SETUP.md](docs/SETUP.md) - Hardware setup and network configuration
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Plugin internals and design

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

Built to connect DGX Spark workstations in ways that go beyond standard configurations. Currently running a 4-node ring cluster with 200Gbps links for distributed LLM training and inference.

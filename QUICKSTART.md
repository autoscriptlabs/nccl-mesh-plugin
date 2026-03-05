# Quick Start Guide

Get distributed LLM training running on your direct-connect RDMA mesh in 15 minutes.

## Prerequisites

- 3+ nodes with direct RDMA connections (ConnectX-7 NICs recommended)
- Each node pair on a separate subnet (e.g., 192.168.100.x, 192.168.101.x, etc.)
- CUDA-capable GPUs (tested on DGX Spark with Grace Hopper)
- Python 3.10+ with PyTorch, DeepSpeed, and Transformers installed
- SLURM job scheduler (optional but recommended)

**Reference cluster**: 4x DGX Spark in ring topology with 200Gbps QSFP56 links (ConnectX-7, dual-channel).

## Step 1: Build the Plugin

```bash
# On each node (or on shared storage accessible by all nodes)
git clone https://github.com/yourusername/nccl-mesh-plugin.git
cd nccl-mesh-plugin

# Install dependencies (Ubuntu/Debian)
sudo apt-get install libibverbs-dev librdmacm-dev

# Build
make
```

Verify the build:
```bash
ls -la libnccl-net.so  # Should exist and be ~160KB
```

## Step 2: Verify RDMA Connectivity

Test that your nodes can communicate via RDMA:

```bash
# On node B (server)
ib_send_bw -d rocep1s0f0 -x 3

# On node A (client) - replace IP with node B's mesh IP
ib_send_bw -d rocep1s0f0 -x 3 192.168.101.3
```

Expected: ~12 GB/s for 100GbE links, ~24 GB/s for 200GbE dual-channel links.

## Step 3: Configure Environment

Set these environment variables before running distributed jobs:

```bash
# Point to your plugin location
export NCCL_NET_PLUGIN=/path/to/nccl-mesh-plugin/libnccl-net.so
export LD_LIBRARY_PATH=/path/to/nccl-mesh-plugin:$LD_LIBRARY_PATH

# Use your management network interface for bootstrapping
export NCCL_SOCKET_IFNAME=eth0  # Adjust to your interface name

# Debug output (optional, use WARN for less verbose)
export NCCL_DEBUG=INFO
```

## Step 4: Test Basic Communication

Create a simple test script (`test_nccl.py`):

```python
import os
import torch
import torch.distributed as dist

# Get rank from environment
rank = int(os.environ.get('RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))
master_addr = os.environ.get('MASTER_ADDR', 'localhost')

# Initialize process group
dist.init_process_group(
    backend='nccl',
    init_method=f'tcp://{master_addr}:29500',
    rank=rank,
    world_size=world_size
)

# Create tensor and all-reduce
tensor = torch.ones(1000, device='cuda') * (rank + 1)
print(f"Rank {rank} before: {tensor[0].item()}")

dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print(f"Rank {rank} after: {tensor[0].item()}")  # Should be sum of all ranks

dist.destroy_process_group()
```

Run on your cluster (e.g., 4 nodes in ring topology):
```bash
# Using SLURM
srun -N4 --ntasks-per-node=1 python test_nccl.py

# Or manually on each node
# Node A: RANK=0 WORLD_SIZE=4 MASTER_ADDR=nodeA python test_nccl.py
# Node B: RANK=1 WORLD_SIZE=4 MASTER_ADDR=nodeA python test_nccl.py
# Node C: RANK=2 WORLD_SIZE=4 MASTER_ADDR=nodeA python test_nccl.py
# Node D: RANK=3 WORLD_SIZE=4 MASTER_ADDR=nodeA python test_nccl.py
```

Look for `NET/Plugin: Loaded net plugin Mesh (v9)` in the output.

## Step 5: Run LLM Training

### Option A: Interactive (for testing)

```bash
# Allocate nodes interactively
salloc -N4 --exclusive

# Run training (100 steps for quick test)
./examples/run_qwen14b_deepspeed.sh --steps 100
```

### Option B: Batch Job (for full training)

```bash
# Submit batch job (defaults to 12000 steps)
sbatch examples/submit_training.sbatch

# Or with custom parameters: steps, warmup, learning_rate
sbatch examples/submit_training.sbatch 12000 100 2e-5

# Monitor progress
tail -f training_<jobid>.log
```

## Configuration Reference

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NCCL_NET_PLUGIN` | Path to mesh plugin | `/path/to/libnccl-net.so` |
| `NCCL_SOCKET_IFNAME` | Bootstrap network interface | `eth0`, `enP7s7` |
| `NCCL_DEBUG` | Log verbosity | `INFO`, `WARN` |
| `NCCL_MESH_GID_INDEX` | RoCE GID index (try 0-3 if issues) | `3` |
| `NCCL_MESH_DEBUG` | Plugin debug level | `0`, `1`, `2` |

### Training Script Options

```bash
./examples/run_qwen14b_deepspeed.sh \
    --steps 12000           # Total training steps
    --warmup 100            # Warmup steps
    --lr 2e-5               # Learning rate
```

### DeepSpeed ZeRO-3 Settings

The training script uses DeepSpeed ZeRO-3 with:
- **Parameter sharding**: Model weights distributed across all nodes
- **Gradient sharding**: Gradients partitioned for memory efficiency
- **Optimizer state sharding**: Adam states split across nodes
- **Gradient checkpointing**: Recompute activations to save memory
- **BF16 training**: Mixed precision with bfloat16

Memory usage for Qwen2.5-14B across 4 nodes: ~29GB allocated per node (less per-node with 4-way sharding vs 3-way).

## Troubleshooting

### Plugin not loading
```
NET/Plugin: Could not find: /path/to/libnccl-net.so
```
- Check the path exists on ALL nodes
- Ensure the plugin was built successfully (`make clean && make`)

### Connection timeout (192.168.x.x addresses)
```
Call to ibv_modify_qp failed with 110 Connection timed out
local GID ::ffff:192.168.100.1, remote GID ::ffff:192.168.101.3
```
- Plugin didn't load; NCCL fell back to built-in IB plugin
- Verify `NCCL_NET_PLUGIN` is set correctly
- Check for `Loaded net plugin Mesh` in logs

### Wrong GID index
```
QP transition to RTR failed
```
- Try different GID indices: `export NCCL_MESH_GID_INDEX=0` (or 1, 2, 3)
- Check available GIDs with `show_gids`

### FlashAttention kernel errors
```
FATAL: Kernel requirements sm80-sm100 not met by sm121
```
- Your GPU compute capability isn't supported by pre-built FA kernels
- Use `attn_implementation="eager"` in the training script (uses more memory)
- Or build FlashAttention from source for your GPU

### Out of memory
- Reduce `max_seq_length` (512 -> 256)
- Reduce `batch_size` (already at 1)
- Enable CPU offloading in DeepSpeed config (slower but uses less GPU memory)

## Using with vLLM

The plugin supports vLLM for distributed inference, but vLLM has a race condition in its NCCL initialization that causes hangs with custom network plugins. Apply the included patch:

```bash
cd /path/to/vllm
git apply /path/to/nccl-mesh-plugin/patches/ticket-f-vllm-nccl-init-barrier.patch
export VLLM_NCCL_INIT_DELAY=2.0
```

See the main [README](README.md#vllm-support) for details.

## What's Next?

- **Scale up**: Ring topology supports 4+ nodes with only 2 NICs each
- **Train larger models**: 32B models work with 4 nodes, 70B needs 6-8 nodes
- **Run inference**: Use vLLM with the included upstream patch
- **Custom datasets**: Modify `SimpleDataset` class in training script

See [docs/SETUP.md](docs/SETUP.md) for detailed hardware setup and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for plugin internals.

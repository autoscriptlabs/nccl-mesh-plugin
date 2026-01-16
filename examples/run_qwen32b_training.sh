#!/bin/bash
#
# Qwen2.5-32B optimized training launcher for 3-node Spark cluster
#
# Memory optimizations:
# - 8-bit Adam: 256GB → 128GB optimizer states
# - Gradient checkpointing: ~50% activation memory reduction
# - Micro-batch=1 with gradient accumulation: Controlled peak memory
#
# Memory budget:
#   Model weights (BF16):     64GB
#   8-bit optimizer states: 128GB (was 256GB with FP32 Adam)
#   Gradients (BF16):         64GB
#   Activations (with ckpt): ~30GB
#   --------------------------------
#   Total:                  ~286GB (fits in 351GB with 65GB headroom)
#

set -e

# Cluster configuration
HEAD_NODE="${HEAD_NODE:-spark-a}"
NODES="${NODES:-spark-a,spark-b,spark-c}"
NUM_NODES="${NUM_NODES:-3}"

# NCCL mesh plugin configuration
export NCCL_NET_PLUGIN=mesh
export NCCL_DEBUG=INFO
export NCCL_MESH_DEBUG=1
export NCCL_SOCKET_IFNAME=enP7s7  # Primary mesh interface

# Triton cache (avoid NFS issues mentioned in your logs)
export TRITON_CACHE_DIR=/tmp/triton_cache_$$

# Disable memory-hungry features
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_qwen32b_optimized.py"

# Parse arguments
DRY_RUN=""
MAX_STEPS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --steps)
            MAX_STEPS="--steps $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "QWEN-32B TRAINING LAUNCHER"
echo "=============================================="
echo "Head node: ${HEAD_NODE}"
echo "Nodes: ${NODES}"
echo "NCCL plugin: mesh"
echo "=============================================="

# Check if running under SLURM
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running under SLURM (job $SLURM_JOB_ID)"

    srun --nodes=${NUM_NODES} \
         --ntasks-per-node=1 \
         --cpus-per-task=64 \
         torchrun \
            --nproc_per_node=1 \
            --nnodes=${NUM_NODES} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${HEAD_NODE}:29500 \
            ${TRAIN_SCRIPT} ${DRY_RUN} ${MAX_STEPS}
else
    echo "Not running under SLURM - use sbatch or srun"
    echo ""
    echo "Example:"
    echo "  salloc -N3 --exclusive bash -c './run_qwen32b_training.sh'"
    echo ""
    echo "Or submit as job:"
    echo "  sbatch --nodes=3 --ntasks-per-node=1 ./run_qwen32b_training.sh"
    exit 1
fi

#!/bin/bash
#
# Qwen2.5-14B training with DeepSpeed ZeRO-3 on 3-node Spark cluster
#

set -e

# Cluster configuration
HEAD_NODE="${HEAD_NODE:-spark-a}"
NODES="${NODES:-spark-a,spark-b,spark-c}"
NUM_NODES="${NUM_NODES:-3}"

# NCCL mesh plugin configuration
export NCCL_NET_PLUGIN=/home/titanic/nccl-mesh-plugin/libnccl-net.so
export LD_LIBRARY_PATH=/home/titanic/nccl-mesh-plugin:${LD_LIBRARY_PATH:-}
export NCCL_DEBUG=INFO
export NCCL_MESH_DEBUG=1
export NCCL_SOCKET_IFNAME=enP7s7

# Triton cache
export TRITON_CACHE_DIR=/tmp/triton_cache_$$

# Memory settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_qwen14b_deepspeed.py"

# Parse arguments
MAX_STEPS=100
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "QWEN-14B DEEPSPEED ZERO-3 TRAINING"
echo "=============================================="
echo "Head node: ${HEAD_NODE}"
echo "Nodes: ${NODES}"
echo "Max steps: ${MAX_STEPS}"
echo "NCCL plugin: mesh"
echo "=============================================="

# Check if running under SLURM
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running under SLURM (job $SLURM_JOB_ID)"

    # Create hostfile for DeepSpeed
    HOSTFILE="/tmp/deepspeed_hostfile_$$"
    scontrol show hostnames $SLURM_JOB_NODELIST | while read host; do
        echo "$host slots=1"
    done > $HOSTFILE

    echo "Hostfile created at $HOSTFILE:"
    cat $HOSTFILE

    # Use srun to launch DeepSpeed on all nodes
    srun --nodes=${NUM_NODES} \
         --ntasks-per-node=1 \
         --cpus-per-task=12 \
         --export=ALL \
         bash -c "echo \"[\$(hostname)] Starting DeepSpeed...\" && \
                  export LD_LIBRARY_PATH=/home/titanic/nccl-mesh-plugin:\${LD_LIBRARY_PATH:-} && \
                  export NCCL_NET_PLUGIN=/home/titanic/nccl-mesh-plugin/libnccl-net.so && \
                  export NCCL_DEBUG=INFO && \
                  export NCCL_MESH_DEBUG=1 && \
                  export NCCL_SOCKET_IFNAME=enP7s7 && \
                  export MASTER_ADDR=${HEAD_NODE} && \
                  export MASTER_PORT=29500 && \
                  export WORLD_SIZE=${NUM_NODES} && \
                  export RANK=\${SLURM_PROCID} && \
                  export LOCAL_RANK=0 && \
                  python ${TRAIN_SCRIPT} \
                     --max_steps ${MAX_STEPS} \
                     --max_seq_length 512 \
                     --batch_size 1 \
                     --gradient_accumulation_steps 16"

    rm -f $HOSTFILE
else
    echo "Not running under SLURM - use sbatch or srun"
    echo ""
    echo "Example:"
    echo "  salloc -N3 --exclusive ./examples/run_qwen14b_deepspeed.sh --steps 100"
    exit 1
fi

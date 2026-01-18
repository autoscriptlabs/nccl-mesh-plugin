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
WARMUP_STEPS=100
LEARNING_RATE="2e-5"
MAX_SEQ_LENGTH=512
BATCH_SIZE=1
GRAD_ACCUM=16
LOCAL_CHECKPOINT_DIR="/tmp/checkpoints"
NAS_CHECKPOINT_DIR="/mnt/nas/titanic/checkpoints"
SAVE_STEPS=500
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --seq-length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --local-checkpoint-dir)
            LOCAL_CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --nas-checkpoint-dir)
            NAS_CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --save-steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --steps N          Maximum training steps (default: 100)"
            echo "  --warmup N         Warmup steps (default: 100)"
            echo "  --lr RATE          Learning rate (default: 2e-5)"
            echo "  --seq-length N     Max sequence length (default: 512)"
            echo "  --batch-size N     Batch size per GPU (default: 1)"
            echo "  --grad-accum N     Gradient accumulation steps (default: 16)"
            echo "  --local-checkpoint-dir  Fast local disk for checkpoints (default: /tmp/checkpoints)"
            echo "  --nas-checkpoint-dir    NAS for durable storage (default: /mnt/nas/titanic/checkpoints)"
            echo "  --save-steps N     Save checkpoint every N steps (default: 500)"
            echo "  --resume PATH      Resume from checkpoint (use 'latest' for auto-detect)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
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
echo "Warmup steps: ${WARMUP_STEPS}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Sequence length: ${MAX_SEQ_LENGTH}"
echo "Batch size: ${BATCH_SIZE}"
echo "Grad accumulation: ${GRAD_ACCUM}"
echo "Local checkpoints: ${LOCAL_CHECKPOINT_DIR} (fast)"
echo "NAS checkpoints: ${NAS_CHECKPOINT_DIR} (durable)"
echo "Save every: ${SAVE_STEPS} steps"
if [ -n "$RESUME" ]; then
    echo "Resume from: ${RESUME}"
fi
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
                     --warmup_steps ${WARMUP_STEPS} \
                     --learning_rate ${LEARNING_RATE} \
                     --max_seq_length ${MAX_SEQ_LENGTH} \
                     --batch_size ${BATCH_SIZE} \
                     --gradient_accumulation_steps ${GRAD_ACCUM} \
                     --local_checkpoint_dir ${LOCAL_CHECKPOINT_DIR} \
                     --nas_checkpoint_dir ${NAS_CHECKPOINT_DIR} \
                     --save_steps ${SAVE_STEPS} \
                     ${RESUME:+--resume_from_checkpoint ${RESUME}}"

    rm -f $HOSTFILE
else
    echo "Not running under SLURM - use sbatch or srun"
    echo ""
    echo "Example:"
    echo "  salloc -N3 --exclusive ./examples/run_qwen14b_deepspeed.sh --steps 100"
    exit 1
fi

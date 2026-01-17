#!/usr/bin/env python3
"""
Optimized Qwen2.5-14B training for 3-node DGX Spark cluster (unified memory)

Memory budget per node: 117GB
Total cluster memory: 351GB

14B model requirements (much more comfortable than 32B):
- Model weights (BF16):     28GB total (~9GB/node with FSDP)
- 8-bit optimizer states:   28GB total (~9GB/node)
- Gradients (BF16):         28GB total (~9GB/node)
- Activations (with ckpt): ~20GB/node
- Total per node:          ~50GB (plenty of headroom)

Usage:
    srun -N3 --ntasks-per-node=1 torchrun --nproc_per_node=1 \
        --nnodes=3 --rdzv_backend=c10d --rdzv_endpoint=$HEAD:29500 \
        train_qwen14b_optimized.py

Environment:
    export NCCL_NET_PLUGIN=mesh
    export NCCL_DEBUG=INFO
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    CPUOffload,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

# 8-bit Adam from bitsandbytes
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("WARNING: bitsandbytes not installed, falling back to standard AdamW")


def get_model_config():
    """Configuration for 14B model on 3x117GB nodes."""
    return {
        # Model - 14B is much more comfortable on 351GB cluster
        "model_id": "/mnt/nas/public/models/Qwen/Qwen2.5-Coder-14B-Instruct",
        "torch_dtype": torch.bfloat16,

        # Training - conservative settings to avoid OOM during FSDP gather
        "micro_batch_size": 1,           # Reduced from 2 to avoid OOM
        "gradient_accumulation_steps": 16,  # Increased to maintain effective batch = 1 * 3 * 16 = 48
        "max_seq_length": 1024,          # Reduced from 2048 to save activation memory
        "learning_rate": 2e-5,
        "warmup_ratio": 0.03,
        "num_epochs": 1,
        "max_grad_norm": 1.0,

        # Memory optimizations (still use them for safety margin)
        "gradient_checkpointing": True,
        "use_8bit_adam": True,

        # Dataset
        "dataset_path": "/mnt/nas/datasets/merged_train.jsonl",
    }


def setup_fsdp_plugin():
    """Configure FSDP for memory efficiency on unified memory systems."""

    # Mixed precision policy - BF16 compute, FP32 reduce for stability
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,  # FP32 for allreduce stability
        buffer_dtype=torch.bfloat16,
    )

    # NOTE: CPUOffload is useless on unified memory - don't enable it
    # It just moves data around in the same memory pool

    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Shard everything
        mixed_precision_policy=mixed_precision_policy,
        cpu_offload=CPUOffload(offload_params=False),   # No CPU offload on unified memory
        use_orig_params=True,  # Required for 8-bit optimizer compatibility
        sync_module_states=True,
        limit_all_gathers=True,  # Memory optimization: limit concurrent allgathers
    )

    return fsdp_plugin


def load_model_with_checkpointing(model_id, dtype):
    """Load model with gradient checkpointing enabled."""

    print(f"Loading model from {model_id}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        use_cache=False,  # Disable KV cache for training
    )

    # Enable gradient checkpointing - critical for memory!
    # This recomputes activations during backward pass instead of storing them
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")

    return model


def get_optimizer(model, config):
    """Get 8-bit Adam optimizer for memory efficiency."""

    lr = config["learning_rate"]

    if config["use_8bit_adam"] and HAS_BNB:
        # 8-bit Adam: momentum and variance stored in 8-bit
        # Reduces optimizer memory from 112GB → 56GB for 14B model
        print("Using 8-bit Adam (bitsandbytes)")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
        )
    else:
        # Standard AdamW - uses 2x more memory for optimizer states
        print("Using standard AdamW (8-bit not available)")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
        )

    return optimizer


def load_dataset_from_jsonl(path, tokenizer, max_length):
    """Load and tokenize dataset from JSONL file."""

    from torch.utils.data import Dataset
    import json

    class JSONLDataset(Dataset):
        def __init__(self, path, tokenizer, max_length):
            self.samples = []
            self.tokenizer = tokenizer
            self.max_length = max_length

            print(f"Loading dataset from {path}...")
            with open(path, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line))
            print(f"Loaded {len(self.samples)} samples")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]

            # SQL dataset format: question, query, db_id, context, source
            context = sample.get('context', '')
            question = sample.get('question', '')
            query = sample.get('query', '')

            # Format as training prompt
            text = f"""<schema>
{context}
</schema>

Question: {question}

SQL: {query}"""

            # Tokenize
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
            )

            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)

            # For causal LM, labels = input_ids (shifted internally by model)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': input_ids.clone(),
            }

    return JSONLDataset(path, tokenizer, max_length)


def print_memory_stats(prefix=""):
    """Print current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"{prefix}Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Load model only, don't train")
    parser.add_argument("--steps", type=int, default=None, help="Max training steps (for testing)")
    args = parser.parse_args()

    config = get_model_config()

    # Setup FSDP-enabled accelerator
    fsdp_plugin = setup_fsdp_plugin()
    accelerator = Accelerator(
        fsdp_plugin=fsdp_plugin,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision="bf16",
    )

    if accelerator.is_main_process:
        print("=" * 60)
        print("OPTIMIZED QWEN-14B TRAINING")
        print("=" * 60)
        print(f"Nodes: {accelerator.num_processes}")
        print(f"Micro-batch size: {config['micro_batch_size']}")
        print(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
        print(f"Effective batch size: {config['micro_batch_size'] * accelerator.num_processes * config['gradient_accumulation_steps']}")
        print(f"8-bit Adam: {config['use_8bit_adam'] and HAS_BNB}")
        print(f"Gradient checkpointing: {config['gradient_checkpointing']}")
        print("=" * 60)
        print_memory_stats("Initial ")

    # Verify model path exists (newer huggingface_hub has strict validation)
    model_path = config["model_id"]
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with gradient checkpointing
    model = load_model_with_checkpointing(
        model_path,
        config["torch_dtype"],
    )

    if accelerator.is_main_process:
        print_memory_stats("After model load ")

    # Get optimizer BEFORE accelerator.prepare() for 8-bit Adam compatibility
    optimizer = get_optimizer(model, config)

    # Load dataset
    dataset = load_dataset_from_jsonl(
        config["dataset_path"],
        tokenizer,
        config["max_seq_length"],
    )

    # Create dataloader with distributed sampler
    dataloader = DataLoader(
        dataset,
        batch_size=config["micro_batch_size"],
        shuffle=False,  # Sampler handles shuffling
        sampler=DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
        ),
        num_workers=0,  # Unified memory - no benefit from multiprocessing
        pin_memory=False,  # No pinning needed on unified memory
    )

    # Calculate training steps
    total_steps = len(dataloader) * config["num_epochs"] // config["gradient_accumulation_steps"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Prepare with accelerator (applies FSDP wrapping)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    if accelerator.is_main_process:
        print_memory_stats("After FSDP wrap ")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")

    if args.dry_run:
        if accelerator.is_main_process:
            print("\nDry run complete - model loaded successfully")
            print_memory_stats("Final ")
        return

    # Training loop
    model.train()
    global_step = 0

    max_steps = args.steps if args.steps else total_steps

    if accelerator.is_main_process:
        print(f"\nStarting training for {max_steps} steps...")

    # Clear memory before training starts
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if accelerator.is_main_process:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"After memory cleanup: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    for epoch in range(config["num_epochs"]):
        # Accelerate may replace our DistributedSampler, so check for set_epoch
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                if accelerator.is_main_process and global_step % 10 == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"Step {global_step}/{max_steps} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
                    print_memory_stats("  ")

                if global_step >= max_steps:
                    break

        if global_step >= max_steps:
            break

    if accelerator.is_main_process:
        print("\nTraining complete!")
        print_memory_stats("Final ")


if __name__ == "__main__":
    main()

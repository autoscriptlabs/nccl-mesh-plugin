#!/usr/bin/env python3
"""
Qwen2.5-14B training with DeepSpeed ZeRO-3 for 3-node DGX Spark cluster

DeepSpeed ZeRO-3 partitions parameters, gradients, and optimizer states
across all nodes, with better memory efficiency than FSDP.

Usage:
    deepspeed --num_nodes=3 --num_gpus=1 --hostfile=hostfile \
        train_qwen14b_deepspeed.py --deepspeed ds_config.json
"""

import os
import argparse
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import deepspeed
from deepspeed import comm as dist


class SimpleDataset(Dataset):
    """Simple dataset from JSONL file."""

    def __init__(self, path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        print(f"Loading dataset from {path}...")
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        # Handle different formats
                        if 'text' in item:
                            self.samples.append(item['text'])
                        elif 'content' in item:
                            self.samples.append(item['content'])
                        elif 'prompt' in item and 'completion' in item:
                            self.samples.append(item['prompt'] + item['completion'])
                        elif 'instruction' in item:
                            text = item['instruction']
                            if 'input' in item and item['input']:
                                text += "\n" + item['input']
                            if 'output' in item:
                                text += "\n" + item['output']
                            self.samples.append(text)
                    except json.JSONDecodeError:
                        continue

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def get_ds_config(args):
    """Generate DeepSpeed ZeRO-3 configuration."""
    return {
        "train_batch_size": args.batch_size * args.gradient_accumulation_steps * args.world_size,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.warmup_steps
            }
        },

        "bf16": {
            "enabled": True
        },

        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },

        "gradient_clipping": args.max_grad_norm,

        "steps_per_print": 10,

        "wall_clock_breakdown": False,

        "zero_allow_untested_optimizer": True
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/mnt/nas/public/models/Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--dataset_path", type=str,
                        default="/mnt/nas/datasets/merged_train.jsonl")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    args = parser.parse_args()

    # Initialize distributed
    deepspeed.init_distributed()
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    if args.rank == 0:
        print("=" * 60)
        print("QWEN-14B TRAINING WITH DEEPSPEED ZERO-3")
        print("=" * 60)
        print(f"World size: {args.world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * args.world_size}")
        print(f"Max sequence length: {args.max_seq_length}")
        print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with DeepSpeed ZeRO-3 init
    ds_config = get_ds_config(args)

    if args.rank == 0:
        print(f"Loading model from {args.model_path}...")

    # Use DeepSpeed's zero.Init for memory-efficient loading
    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False,  # Required for gradient checkpointing
        )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    if args.rank == 0:
        print("Model loaded with gradient checkpointing enabled")
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    # Load dataset
    dataset = SimpleDataset(args.dataset_path, tokenizer, args.max_seq_length)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )

    if args.rank == 0:
        print(f"\nStarting training for {args.max_steps} steps...")
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"After DeepSpeed init: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    # Training loop
    model_engine.train()
    global_step = 0
    total_loss = 0.0

    for batch in dataloader:
        if global_step >= args.max_steps:
            break

        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        # Backward pass (DeepSpeed handles gradient accumulation)
        model_engine.backward(loss)
        model_engine.step()

        total_loss += loss.item()
        global_step += 1

        # Log progress
        if global_step % 10 == 0 and args.rank == 0:
            avg_loss = total_loss / 10
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"Step {global_step}/{args.max_steps} | Loss: {avg_loss:.4f} | "
                  f"Memory: {allocated:.1f}GB/{reserved:.1f}GB")
            total_loss = 0.0

    if args.rank == 0:
        print("\nTraining complete!")


if __name__ == "__main__":
    main()

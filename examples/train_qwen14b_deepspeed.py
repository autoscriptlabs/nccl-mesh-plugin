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
import shutil
import threading
import time
from queue import Queue
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import deepspeed
from deepspeed import comm as dist


class AsyncCheckpointManager:
    """
    Manages async checkpoint copying from local disk to NAS.

    Workflow:
    1. Save checkpoint to fast local disk (blocking, but fast)
    2. Background thread copies to NAS (non-blocking)
    3. Once NAS copy verified, delete local copy
    4. Training continues uninterrupted
    """

    def __init__(self, local_dir, nas_dir, rank=0):
        self.local_dir = local_dir
        self.nas_dir = nas_dir
        self.rank = rank
        self.copy_queue = Queue()
        self.worker_thread = None
        self.shutdown_flag = threading.Event()
        self.pending_copies = []

        # Create directories
        if rank == 0:
            os.makedirs(local_dir, exist_ok=True)
            os.makedirs(nas_dir, exist_ok=True)

        # Start background worker
        self._start_worker()

    def _start_worker(self):
        """Start background copy worker thread."""
        self.worker_thread = threading.Thread(target=self._copy_worker, daemon=True)
        self.worker_thread.start()

    def _copy_worker(self):
        """Background worker that copies checkpoints to NAS."""
        while not self.shutdown_flag.is_set():
            try:
                # Wait for work with timeout (allows checking shutdown flag)
                try:
                    local_path, nas_path, step = self.copy_queue.get(timeout=1.0)
                except:
                    continue

                if self.rank == 0:
                    print(f"[Checkpoint] Starting background copy of step {step} to NAS...", flush=True)

                try:
                    # Copy directory tree to NAS
                    start_time = time.time()
                    if os.path.exists(nas_path):
                        shutil.rmtree(nas_path)
                    shutil.copytree(local_path, nas_path)
                    copy_time = time.time() - start_time

                    # Verify copy by checking file count and total size
                    local_size = self._get_dir_size(local_path)
                    nas_size = self._get_dir_size(nas_path)

                    if abs(local_size - nas_size) < 1024:  # Allow 1KB variance
                        # Copy verified, safe to delete local
                        if self.rank == 0:
                            print(f"[Checkpoint] NAS copy verified ({nas_size/1e9:.1f}GB in {copy_time:.1f}s). Removing local copy.", flush=True)
                        shutil.rmtree(local_path)
                        self.pending_copies.remove(step)
                    else:
                        if self.rank == 0:
                            print(f"[Checkpoint] WARNING: Size mismatch! Local={local_size}, NAS={nas_size}. Keeping local copy.", flush=True)

                except Exception as e:
                    if self.rank == 0:
                        print(f"[Checkpoint] ERROR copying to NAS: {e}. Local copy preserved.", flush=True)

                self.copy_queue.task_done()

            except Exception as e:
                if self.rank == 0:
                    print(f"[Checkpoint] Worker error: {e}", flush=True)

    def _get_dir_size(self, path):
        """Get total size of directory in bytes."""
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total

    def get_local_path(self, name="latest"):
        """Get local checkpoint path."""
        return os.path.join(self.local_dir, name)

    def queue_nas_copy(self, name, step):
        """Queue a checkpoint for async copy to NAS."""
        local_path = os.path.join(self.local_dir, name)
        nas_path = os.path.join(self.nas_dir, name)
        self.pending_copies.append(step)
        self.copy_queue.put((local_path, nas_path, step))
        if self.rank == 0:
            print(f"[Checkpoint] Queued step {step} for NAS copy (training continues).", flush=True)

    def wait_for_copies(self, timeout=300):
        """Wait for all pending copies to complete (call at end of training)."""
        if self.rank == 0 and self.pending_copies:
            print(f"[Checkpoint] Waiting for {len(self.pending_copies)} pending NAS copies...", flush=True)
        self.copy_queue.join()
        if self.rank == 0:
            print("[Checkpoint] All NAS copies complete.", flush=True)

    def shutdown(self):
        """Shutdown the background worker."""
        self.shutdown_flag.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5)


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
                        elif 'question' in item and 'query' in item:
                            # Text2SQL format: question + query
                            text = item['question'] + "\n" + item['query']
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
    parser.add_argument("--local_checkpoint_dir", type=str, default="/tmp/checkpoints",
                        help="Fast local disk for checkpoint writes")
    parser.add_argument("--nas_checkpoint_dir", type=str, default="/mnt/nas/titanic/checkpoints",
                        help="NAS directory for durable checkpoint storage")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from, or 'latest' to auto-detect")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    args = parser.parse_args()

    # Check if model path exists locally
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    # Initialize distributed
    deepspeed.init_distributed()
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    # Initialize async checkpoint manager (saves locally, copies to NAS in background)
    ckpt_manager = AsyncCheckpointManager(
        local_dir=args.local_checkpoint_dir,
        nas_dir=args.nas_checkpoint_dir,
        rank=args.rank
    )

    if args.rank == 0:
        print("=" * 60)
        print("QWEN-14B TRAINING WITH DEEPSPEED ZERO-3")
        print("=" * 60)
        print(f"World size: {args.world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * args.world_size}")
        print(f"Max sequence length: {args.max_seq_length}")
        print(f"Local checkpoint dir: {args.local_checkpoint_dir} (fast)")
        print(f"NAS checkpoint dir: {args.nas_checkpoint_dir} (durable)")
        print(f"Save every: {args.save_steps} steps")
        print("=" * 60)

    # Load tokenizer (local_files_only=True for local paths)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get DeepSpeed config
    ds_config = get_ds_config(args)

    if args.rank == 0:
        print(f"Loading model from {args.model_path}...")

    # Load model with memory optimizations
    # Don't use zero.Init() - it doesn't work well with from_pretrained
    # Instead, use low_cpu_mem_usage to load shards incrementally
    # Note: flash_attention_2 hangs on sm121 GPUs, use eager instead
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        use_cache=False,  # Required for gradient checkpointing
        low_cpu_mem_usage=True,  # Load shards incrementally
        attn_implementation="eager",  # FA2 hangs on sm121
    )

    # Enable gradient checkpointing before DeepSpeed init
    model.gradient_checkpointing_enable()

    if args.rank == 0:
        print("Model loaded with gradient checkpointing enabled")
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    # Load dataset
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {args.dataset_path}")

    dataset = SimpleDataset(args.dataset_path, tokenizer, args.max_seq_length)

    if len(dataset) == 0:
        raise ValueError(
            f"No samples loaded from {args.dataset_path}. "
            "Check that the file contains valid JSONL with 'text', 'content', "
            "'prompt'+'completion', 'instruction', or 'question'+'query' fields."
        )

    if args.rank == 0:
        print(f"Dataset loaded with {len(dataset)} samples")

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

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            # Check NAS first (durable), then local
            nas_latest = os.path.join(args.nas_checkpoint_dir, "latest")
            local_latest = os.path.join(args.local_checkpoint_dir, "latest")

            if os.path.exists(nas_latest):
                args.resume_from_checkpoint = nas_latest
                if args.rank == 0:
                    print(f"Found checkpoint on NAS: {nas_latest}", flush=True)
            elif os.path.exists(local_latest):
                args.resume_from_checkpoint = local_latest
                if args.rank == 0:
                    print(f"Found checkpoint on local disk: {local_latest}", flush=True)
            else:
                args.resume_from_checkpoint = None
                if args.rank == 0:
                    print("No checkpoint found to resume from, starting fresh.", flush=True)

        if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
            if args.rank == 0:
                print(f"Resuming from checkpoint: {args.resume_from_checkpoint}", flush=True)
            _, client_state = model_engine.load_checkpoint(args.resume_from_checkpoint)
            if client_state and 'step' in client_state:
                start_step = client_state['step']
                if args.rank == 0:
                    print(f"Resumed at step {start_step}", flush=True)

    if args.rank == 0:
        print(f"\nStarting training for {args.max_steps} steps (from step {start_step})...", flush=True)
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"After DeepSpeed init: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved", flush=True)

    # Training loop
    model_engine.train()
    global_step = start_step
    total_loss = 0.0

    if args.rank == 0:
        print("Starting first batch...", flush=True)

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

        # Log progress - every step for first 5, then every 10
        if args.rank == 0 and (global_step <= 5 or global_step % 10 == 0):
            if global_step <= 5:
                # Detailed logging for first few steps
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"Step {global_step}/{args.max_steps} | Loss: {loss.item():.4f} | "
                      f"Memory: {allocated:.1f}GB/{reserved:.1f}GB", flush=True)
            else:
                avg_loss = total_loss / 10
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"Step {global_step}/{args.max_steps} | Loss: {avg_loss:.4f} | "
                      f"Memory: {allocated:.1f}GB/{reserved:.1f}GB", flush=True)
                total_loss = 0.0

        # Save checkpoint to local disk, then async copy to NAS
        if global_step > 0 and global_step % args.save_steps == 0:
            local_checkpoint_path = ckpt_manager.get_local_path("latest")
            if args.rank == 0:
                print(f"Saving checkpoint at step {global_step} to local disk...", flush=True)
            model_engine.save_checkpoint(local_checkpoint_path, client_state={'step': global_step})
            if args.rank == 0:
                print(f"Checkpoint saved locally. Queuing NAS copy...", flush=True)
            # Queue async copy to NAS (training continues immediately)
            ckpt_manager.queue_nas_copy("latest", global_step)

    # Save final checkpoint to local, then async copy to NAS
    final_local_path = ckpt_manager.get_local_path("final")
    if args.rank == 0:
        print(f"Saving final checkpoint (step {global_step}) to local disk...", flush=True)
    model_engine.save_checkpoint(final_local_path, client_state={'step': global_step})
    ckpt_manager.queue_nas_copy("final", global_step)

    # Wait for all NAS copies to complete before exiting
    ckpt_manager.wait_for_copies()
    ckpt_manager.shutdown()

    if args.rank == 0:
        print("\nTraining complete!")
        print(f"Final checkpoint available at: {args.nas_checkpoint_dir}/final")


if __name__ == "__main__":
    main()

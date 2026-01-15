#!/usr/bin/env python3
"""Multi-node inference benchmark using FSDP"""
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator


def is_local_path(model_path):
    """Check if the model path is a local filesystem path."""
    return os.path.exists(model_path) or model_path.startswith('/')

def benchmark_model(model_id, accelerator, num_tokens=50, num_runs=3):
    if accelerator.is_main_process:
        print(f"  Loading {model_id}...")

    # Detect local paths for local_files_only parameter
    use_local = is_local_path(model_id)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=use_local,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=use_local,
    )
    
    model = accelerator.prepare(model)
    model.eval()
    
    prompt = "Explain the theory of relativity in simple terms:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(accelerator.device)
    
    # Warmup
    if accelerator.is_main_process:
        print(f"  Warmup...", end=" ", flush=True)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("done")
    
    # Benchmark
    if accelerator.is_main_process:
        print(f"  Benchmarking...", end=" ", flush=True)
    
    times = []
    for _ in range(num_runs):
        accelerator.wait_for_everyone()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=num_tokens, pad_token_id=tokenizer.pad_token_id)
        accelerator.wait_for_everyone()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    tok_per_sec = num_tokens / avg_time
    
    if accelerator.is_main_process:
        print(f"{tok_per_sec:.1f} tok/s")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return tok_per_sec

def main():
    import argparse
    parser = argparse.ArgumentParser(description='FSDP Benchmark')
    parser.add_argument('--model', type=str, default=None,
                       help='Single model path or HuggingFace ID to benchmark')
    parser.add_argument('--num-tokens', type=int, default=50,
                       help='Number of tokens to generate per run')
    parser.add_argument('--num-runs', type=int, default=3,
                       help='Number of benchmark runs')
    args = parser.parse_args()

    accelerator = Accelerator()

    # Default models to benchmark, or use single model from command line
    if args.model:
        model_name = os.path.basename(args.model.rstrip('/')) or "custom"
        models = {model_name: args.model}
    else:
        models = {
            "7B": "Qwen/Qwen2.5-7B-Instruct",
            "14B": "Qwen/Qwen2.5-14B-Instruct",
            "32B": "Qwen/Qwen2.5-32B-Instruct",
            "72B": "Qwen/Qwen2.5-72B-Instruct",
        }

    if accelerator.is_main_process:
        print("=" * 60)
        print(f"DISTRIBUTED BENCHMARK ({accelerator.num_processes} NODES)")
        print("=" * 60)

    results = {}
    for size, model_id in models.items():
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"\n[{size} Model]")
        try:
            results[size] = benchmark_model(model_id, accelerator, args.num_tokens, args.num_runs)
        except Exception as e:
            if accelerator.is_main_process:
                if "out of memory" in str(e).lower():
                    print(f"  OOM!")
                    results[size] = "OOM"
                else:
                    print(f"  Error: {e}")
                    results[size] = "ERROR"
            torch.cuda.empty_cache()
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"{'Model':<30} {'tok/s':<15}")
        print("-" * 45)
        for name, val in results.items():
            if isinstance(val, float):
                print(f"{name:<30} {val:<15.1f}")
            else:
                print(f"{name:<30} {val:<15}")
        print("=" * 60)

if __name__ == "__main__":
    main()

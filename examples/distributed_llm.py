#!/usr/bin/env python3
"""
Distributed LLM Inference with NCCL Mesh Plugin

This example demonstrates loading and running inference on a large language
model distributed across multiple GPUs using the NCCL Mesh Plugin.

Usage:
    # On each node (adjust --rank):
    python distributed_llm.py --rank 0 --world-size 3 --master-ip 10.0.0.170

Environment setup (run on each node):
    cd ~/nccl-mesh-plugin
    export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
    export NCCL_NET_PLUGIN=mesh
    export NCCL_DEBUG=WARN
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator


def is_local_path(model_path):
    """Check if the model path is a local filesystem path."""
    return os.path.exists(model_path) or model_path.startswith('/')


def main():
    parser = argparse.ArgumentParser(description='Distributed LLM Inference')
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, default=3)
    parser.add_argument('--master-ip', type=str, default='10.0.0.170')
    parser.add_argument('--master-port', type=int, default=29500)
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2',
                       help='Model to load (default: Mistral-7B)')
    parser.add_argument('--prompt', type=str, 
                       default='The future of distributed AI computing is',
                       help='Prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum tokens to generate')
    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator()
    
    # Determine if model is a local path or HuggingFace repo ID
    use_local = is_local_path(args.model)
    if use_local:
        print(f'Rank {accelerator.process_index}: Using local model path: {args.model}')

    print(f'Rank {accelerator.process_index}: Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=use_local,
        trust_remote_code=True,
    )

    print(f'Rank {accelerator.process_index}: Loading model...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        local_files_only=use_local,
        trust_remote_code=True,
    )
    
    print(f'Rank {accelerator.process_index}: Model loaded!')

    # Only rank 0 generates
    if accelerator.is_main_process:
        print(f'\nGenerating text...')
        print(f'Prompt: "{args.prompt}"\n')
        
        inputs = tokenizer(args.prompt, return_tensors='pt').to('cuda')
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print('=' * 60)
        print('Generated Text:')
        print('=' * 60)
        print(result)
        print('=' * 60)

    # Wait for all ranks
    accelerator.wait_for_everyone()
    print(f'Rank {accelerator.process_index}: Done!')


if __name__ == '__main__':
    main()

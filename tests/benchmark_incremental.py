#!/usr/bin/env python3
"""
Incremental Distributed Benchmark for NCCL Mesh Plugin.

Writes rank-0 results after every completed test so partial data survives
later failures.
"""

import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist


def log_result(filepath: Path, result: dict) -> None:
    """Append one JSON result and force it to disk."""
    with filepath.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
        f.flush()
        os.fsync(f.fileno())


def current_device() -> torch.device:
    """Return the CUDA device selected by main()."""
    return torch.device("cuda", torch.cuda.current_device())


def get_system_info() -> dict:
    """Gather system information for the rank-0 result header."""
    device_index = torch.cuda.current_device()
    return {
        "hostname": socket.gethostname(),
        "cuda_device": (
            torch.cuda.get_device_name(device_index)
            if torch.cuda.is_available()
            else "N/A"
        ),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "timestamp": datetime.now().isoformat(),
    }


def benchmark_allreduce(size_mb: int, iterations: int = 10, warmup: int = 3) -> dict:
    """Benchmark NCCL all-reduce and report algorithm/bus bandwidth."""
    device = current_device()
    element_count = size_mb * 1024 * 1024 // 4
    payload_bytes = element_count * 4

    data = torch.randn(element_count, device=device)
    torch.cuda.synchronize()

    for _ in range(warmup):
        dist.all_reduce(data)
    torch.cuda.synchronize()

    dist.barrier()
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        dist.all_reduce(data)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    world_size = dist.get_world_size()

    algorithm_bw = payload_bytes / avg_time / 1e9
    bus_bw = algorithm_bw * 2 * (world_size - 1) / world_size
    peak_algorithm_bw = payload_bytes / min_time / 1e9
    peak_bus_bw = peak_algorithm_bw * 2 * (world_size - 1) / world_size

    return {
        "test": "allreduce",
        "size_mb": size_mb,
        "iterations": iterations,
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "algorithm_bandwidth_gbps": algorithm_bw,
        "avg_bandwidth_gbps": bus_bw,
        "peak_bandwidth_gbps": peak_bus_bw,
    }


def benchmark_broadcast(size_mb: int, iterations: int = 10, warmup: int = 3) -> dict:
    """Benchmark broadcast from rank 0."""
    device = current_device()
    element_count = size_mb * 1024 * 1024 // 4
    payload_bytes = element_count * 4

    data = torch.randn(element_count, device=device)
    torch.cuda.synchronize()

    for _ in range(warmup):
        dist.broadcast(data, src=0)
    torch.cuda.synchronize()

    dist.barrier()
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        dist.broadcast(data, src=0)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)

    return {
        "test": "broadcast",
        "size_mb": size_mb,
        "iterations": iterations,
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min(times) * 1000,
        "max_time_ms": max(times) * 1000,
        "avg_bandwidth_gbps": payload_bytes / avg_time / 1e9,
    }


def benchmark_allgather(size_mb: int, iterations: int = 10, warmup: int = 3) -> dict:
    """Benchmark all-gather."""
    device = current_device()
    world_size = dist.get_world_size()
    element_count = size_mb * 1024 * 1024 // 4
    payload_bytes = element_count * 4

    input_tensor = torch.randn(element_count, device=device)
    output_tensors = [
        torch.empty(element_count, device=device) for _ in range(world_size)
    ]
    torch.cuda.synchronize()

    for _ in range(warmup):
        dist.all_gather(output_tensors, input_tensor)
    torch.cuda.synchronize()

    dist.barrier()
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        dist.all_gather(output_tensors, input_tensor)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    data_moved = payload_bytes * (world_size - 1)

    return {
        "test": "allgather",
        "size_mb": size_mb,
        "iterations": iterations,
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min(times) * 1000,
        "max_time_ms": max(times) * 1000,
        "avg_bandwidth_gbps": data_moved / avg_time / 1e9,
    }


def benchmark_reduce_scatter(
    size_mb: int,
    iterations: int = 10,
    warmup: int = 3,
) -> dict:
    """Benchmark reduce-scatter."""
    device = current_device()
    world_size = dist.get_world_size()
    element_count = size_mb * 1024 * 1024 // 4

    # Input length must be divisible by world_size.
    element_count = (element_count // world_size) * world_size
    payload_bytes = element_count * 4

    input_tensor = torch.randn(element_count, device=device)
    output_tensor = torch.empty(element_count // world_size, device=device)
    torch.cuda.synchronize()

    for _ in range(warmup):
        dist.reduce_scatter_tensor(output_tensor, input_tensor)
    torch.cuda.synchronize()

    dist.barrier()
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        dist.reduce_scatter_tensor(output_tensor, input_tensor)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    data_moved = payload_bytes * (world_size - 1) / world_size

    return {
        "test": "reduce_scatter",
        "size_mb": size_mb,
        "iterations": iterations,
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min(times) * 1000,
        "max_time_ms": max(times) * 1000,
        "avg_bandwidth_gbps": data_moved / avg_time / 1e9,
    }


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    hostname = socket.gethostname()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"benchmark_results_{timestamp}.jsonl")

    try:
        if rank == 0:
            print(f"[{hostname}] Starting benchmark, results -> {output_file}")
            log_result(
                output_file,
                {
                    "type": "header",
                    "world_size": world_size,
                    "system_info": get_system_info(),
                },
            )

        sizes = [1, 10, 50, 100, 250, 500, 1000]
        tests = [
            ("allreduce", benchmark_allreduce),
            ("broadcast", benchmark_broadcast),
            ("allgather", benchmark_allgather),
            ("reduce_scatter", benchmark_reduce_scatter),
        ]

        for test_name, test_fn in tests:
            if rank == 0:
                print(f"\n{'=' * 60}")
                print(f"Running {test_name} benchmarks...")
                print("=" * 60)

            for size_mb in sizes:
                dist.barrier()

                if rank == 0:
                    print(
                        f"  {test_name} @ {size_mb}MB ... ",
                        end="",
                        flush=True,
                    )

                result = test_fn(size_mb, iterations=10, warmup=3)
                result["timestamp"] = datetime.now().isoformat()
                result["status"] = "success"

                if rank == 0:
                    log_result(output_file, result)
                    print(
                        f"{result['avg_bandwidth_gbps']:.2f} GB/s "
                        f"(avg: {result['avg_time_ms']:.2f}ms)"
                    )

                time.sleep(0.5)

        if rank == 0:
            log_result(
                output_file,
                {
                    "type": "footer",
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                },
            )
            print(f"\n{'=' * 60}")
            print(f"Benchmark complete! Results saved to: {output_file}")
            print("=" * 60)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

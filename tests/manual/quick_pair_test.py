#!/usr/bin/env python3
"""One-shot all-reduce bandwidth test between distributed ranks."""

import argparse
import time

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-ip", required=True)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--size-mb", type=int, default=500)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        rank=args.rank,
        world_size=args.world_size,
        init_method=f"tcp://{args.master_ip}:{args.master_port}",
        device_id=device,
    )

    try:
        element_count = args.size_mb * 1024 * 1024 // 4
        payload_bytes = element_count * 4
        data = torch.randn(element_count, device=device)
        torch.cuda.synchronize()

        for _ in range(args.warmup):
            dist.all_reduce(data)
        torch.cuda.synchronize()
        dist.barrier()

        times = []
        for _ in range(args.iters):
            start = time.perf_counter()
            dist.all_reduce(data)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        algorithm_bw = payload_bytes / avg_time / 1e9
        bus_bw = (
            algorithm_bw
            * 2
            * (args.world_size - 1)
            / args.world_size
        )

        if args.rank == 0:
            print(
                f"RESULT size={args.size_mb}MB "
                f"avg_time={avg_time * 1000:.2f}ms "
                f"alg_bw={algorithm_bw:.2f}GB/s "
                f"bus_bw={bus_bw:.2f}GB/s"
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

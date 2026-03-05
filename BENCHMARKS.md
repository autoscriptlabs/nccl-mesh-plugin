# NCCL Mesh Plugin Benchmark Results

> **Note:** These benchmarks were collected on the original 3-node triangle mesh with 100Gbps single-channel links. The cluster has since been upgraded to a **4-node ring topology with 200Gbps QSFP56 dual-channel links**. Updated benchmarks are pending.

## 3-Node Triangle Mesh (100Gbps, January 2026)

- **World Size:** 3
- **Device:** NVIDIA GB10
- **Date:** 2026-01-09
- **Link Speed:** 100 Gbps (single channel per ConnectX-7 port)

## Allreduce

| Size (MB) | Bandwidth (GB/s) | Latency (ms) |
|-----------|------------------|--------------|
| 1 | 0.49 | 4.27 |
| 10 | 7.36 | 2.85 |
| 50 | 11.59 | 9.05 |
| 100 | 15.24 | 13.77 |
| 250 | 16.10 | 32.57 |
| 500 | 16.56 | 63.32 |
| 1000 | 16.93 | 123.89 |

## Broadcast

| Size (MB) | Bandwidth (GB/s) | Latency (ms) |
|-----------|------------------|--------------|
| 1 | 4.50 | 0.23 |
| 10 | 3.70 | 2.84 |
| 50 | 8.57 | 6.12 |
| 100 | 9.85 | 10.64 |
| 250 | 7.28 | 36.01 |
| 500 | 7.80 | 67.19 |
| 1000 | 7.91 | 132.64 |

## Allgather

| Size (MB) | Bandwidth (GB/s) | Latency (ms) |
|-----------|------------------|--------------|
| 1 | 0.85 | 2.48 |
| 10 | 5.08 | 4.13 |
| 50 | 8.34 | 12.57 |
| 100 | 8.51 | 24.64 |
| 250 | 9.23 | 56.82 |
| 500 | 9.76 | 107.42 |
| 1000 | 10.02 | 209.30 |

## Reduce Scatter

| Size (MB) | Bandwidth (GB/s) | Latency (ms) |
|-----------|------------------|--------------|
| 1 | 0.31 | 2.27 |
| 10 | 3.07 | 2.28 |
| 50 | 5.08 | 6.89 |
| 100 | 7.77 | 8.99 |
| 250 | 9.56 | 18.27 |
| 500 | 10.05 | 34.79 |
| 1000 | 10.27 | 68.05 |

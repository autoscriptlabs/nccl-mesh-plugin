# Four-node DGX Spark validation — August 2026

## Purpose

This report records the validation evidence for the Gate 1 and Gate 1.5 changes
on a real four-node direct-attached RoCE ring.

It is a reproducibility record, not a general performance guarantee.

## Source history

```text
2f7371be324cf3101f4436bb105b437dc6230bce
  Fix NCCL ABI and enable system RDMA transport

89ad5a89cd2c640069a53db3ed0b237bbfaa2b1b
  gate1: add typed transport objects and persistent completions

07336ca4427adf795e628a5445ee79fc249500f1
  gate1.5: add opt-in hybrid TCP fallback
```

Validated plugin SHA256 on all four hosts:

```text
717b613b8557f629c0a77161559fd6acd1e9a36b0f5b4dac36458ac64b5ec4c1
```

## Hardware topology

- 4 × NVIDIA DGX Spark
- one GB10 GPU/process per host
- two direct CX-7 RoCE links per host
- physical ring: `spark-a → spark-b → spark-c → spark-d → spark-a`
- separate shared 10 GbE management network
- management interface: `enP7s7`
- Linux/AArch64

Private management addresses used during the run:

| Host | Address |
|---|---|
| spark-a | `10.0.0.170` |
| spark-b | `10.0.0.171` |
| spark-c | `10.0.0.172` |
| spark-d | `10.0.0.173` |

## Software

Host collective regression:

- NCCL 2.29.7+cuda13.2

Production container:

- image ID `sha256:70a22be73c6b5dd23544f4112afb1e84c0934e40a1f1a8096593cd154c9c3ff9`
- vLLM `0.26.1rc1.dev229+g124154a88`
- PyTorch `2.13.0+cu130`
- CUDA 13.0
- explicitly preloaded NCCL 2.30.4 from `/opt/nccl-2.30.4/libnccl.so.2`

## Gate 1 tests

- routing tests: 13/13
- error-path tests: 66/66
- typed object-dispatch tests: pass
- NCCL network handle: 120 bytes, within the 128-byte ABI limit
- build: clean

## Adjacent RDMA regression

A 1,000 MiB adjacent-pair transfer remained on direct RDMA and reached
approximately 13.28 GB/s, matching the pre-Gate-1.5 baseline.

## Forced diagonal hybrid TCP

A non-neighbor `spark-a ↔ spark-c` test was forced under NCCL 2.29.7:

- hybrid TCP selected explicitly;
- 100 MiB transfer;
- approximately 1.16 GB/s;
- nonzero send/receive bytes;
- zero reported connection errors.

The same diagonal test was repeated inside the production container with the
exact NCCL 2.30.4 library:

- `NCCL version 2.30.4+cuda13.2`;
- hybrid TCP selected;
- 100 MiB transfer;
- approximately 1.14 GB/s;
- 16 send and 16 receive channels;
- zero reported connection errors.

## Four-node collectives

Under NCCL 2.29.7, all 28 test cases passed.

Representative 1,000 MiB results:

| Collective | Reported throughput |
|---|---:|
| All-reduce | 13.56 GB/s |
| Broadcast | 13.60 GB/s |
| All-gather | 11.83 GB/s |
| Reduce-scatter | 13.19 GB/s |

A four-node benchmark using the exact production NCCL 2.30.4 library also
completed successfully.

Transport summaries showed:

- bulk workload connections on RDMA;
- diagonal `spark-a ↔ spark-c` TCP connection: 896 bytes in each direction on
  one connection, with other diagonal connections at zero bytes;
- diagonal `spark-b ↔ spark-d` TCP shortcut connections: zero payload bytes;
- zero connection errors;
- final visible reduce-scatter throughput: 13.18 GB/s.

This supports the design assumption that the physical ring carries the payload
and hybrid TCP keeps sparse non-neighbor setup connections functional.

## Exact production vLLM validation

Model:

```text
olka-fi/MiniMax-M3-MXFP4
```

Deployment:

- tensor parallelism: 4
- one worker per Spark
- exact production NCCL 2.30.4
- Mesh plugin ABI v9
- `NCCL_ALGO=Ring`
- `NCCL_MESH_HYBRID_TCP=1`
- multiprocess distributed executor selected by the launch override

The first exact-production launch completed NCCL communicator initialization and
model loading but later stopped making progress. NCCL reported:

```text
transport/net.cc:1538 -> 3
proxy.cc:974 -> 3 [Progress Thread]
```

The deployed binary and hybrid environment were verified on all four hosts, so
a stale plugin was ruled out.

The source-level interpretation was that the network-plugin `test()` operation
returned `ncclInternalError`, causing the NCCL proxy progress thread to exit.
The plugin's exact structured failure line was not captured.

Two relevant settings were changed together for the next launch:

```text
NCCL_MESH_FAST_FAIL=0
NCCL_MESH_TIMEOUT_SEC=900
```

Periodic metrics were disabled because their latency values were unreliable:

```text
NCCL_MESH_METRICS=0
```

The next exact-production launch completed:

- NCCL communicator initialization;
- main-model loading;
- EAGLE3 draft loading;
- KV-cache/runtime initialization;
- engine readiness;
- HTTP service startup;
- successful chat completion.

Because fast-fail and operation timeout changed together, the recovery cannot be
attributed to only one setting.

## End-to-end benchmark

llama-benchy 0.4.0 ran:

- prompt size: 2,048 tokens
- generated size: 128 tokens
- context depths: 0, 4,096, 16,384
- concurrency: 1, 2, 4
- 3 runs per matrix point
- 27 measured runs total

All nine matrix tasks completed.

The table below uses the non-prefill-phase records reported in the benchmark
summary:

| Depth | Concurrency | Prefill tok/s | Aggregate decode tok/s | Per-request decode tok/s | TTFR ms |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1774.58 | 35.87 | 35.87 | 1155.84 |
| 0 | 2 | 1589.71 | 42.21 | 25.07 | 1899.26 |
| 0 | 4 | 1938.62 | 54.33 | 17.71 | 3159.74 |
| 4096 | 1 | 1552.77 | 33.13 | 33.13 | 1319.72 |
| 4096 | 2 | 1556.14 | 48.06 | 26.50 | 2245.43 |
| 4096 | 4 | 1660.28 | 50.54 | 17.49 | 3722.72 |
| 16384 | 1 | 1496.21 | 31.57 | 31.57 | 1369.86 |
| 16384 | 2 | 1505.18 | 40.17 | 23.58 | 2037.70 |
| 16384 | 4 | 1600.09 | 47.04 | 16.22 | 3781.64 |

`tg_throughput` is aggregate server decode throughput.
`tg_req_throughput` is average per-request decode throughput.

For context depths above zero, llama-benchy also emitted separate
`is_context_prefill_phase=true` records. Those records are not mixed into the
steady-state continuation table above.

## Gate verdict

| Gate | Result |
|---|---|
| Build and unit tests | PASS |
| Adjacent direct RDMA | PASS |
| Forced diagonal hybrid TCP | PASS |
| Four-node NCCL 2.29.7 collectives | PASS |
| Four-node exact NCCL 2.30.4 collectives | PASS |
| Exact production vLLM TP=4 startup | PASS |
| End-to-end inference | PASS |
| 27-run performance matrix | PASS |

Gate 1.5 is approved as an opt-in hybrid transport.

## Known follow-up work

- Repair periodic latency accounting.
- Document the production timeout profile prominently.
- Isolate `NCCL_MESH_FAST_FAIL` versus `NCCL_MESH_TIMEOUT_SEC` in a controlled
  A/B test only when operationally convenient.
- Evaluate draft-model acceptance separately from transport work.
- Consider full RDMA relay only if future measurements show non-neighbor TCP
  carrying material payload.

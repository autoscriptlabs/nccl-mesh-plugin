# NCCL Mesh Plugin

**A practical NCCL network plugin for small direct-attached RoCE clusters that
cannot justify a switched RDMA fabric.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Operational](https://img.shields.io/badge/status-operational-brightgreen.svg)](#project-status)

## Overview

NCCL Mesh makes point-to-point RoCE links usable as a real multi-node NCCL
transport. Each cable can live on its own IPv4 subnet. The plugin advertises all
usable local RDMA addresses and chooses the local NIC whose subnet matches the
peer address for each connection.

For sparse topologies, an opt-in hybrid mode keeps directly reachable peers on
RDMA and uses the shared management network only for peer pairs that do not share
an RDMA subnet:

| Peer relationship | `NCCL_MESH_HYBRID_TCP=0` | `NCCL_MESH_HYBRID_TCP=1` |
|---|---|---|
| Shared RDMA subnet | Direct RDMA | Direct RDMA |
| No shared RDMA subnet | Connection fails | Direct management TCP |

Hybrid TCP is a per-connection fallback. It is **not** store-and-forward relay
through another GPU node.

## Why this exists

A switched fabric remains the most flexible design, but the switch can cost as
much as the compute in a small self-funded cluster. A direct-attached ring uses
the CX-7 ports already present in the systems:

```text
spark-a ───── spark-b
   │             │
   │             │
spark-d ───── spark-c
```

Each node has one direct RoCE link to each neighbor and a separate, all-reachable
management network for rendezvous, NCCL bootstrap, and optional non-neighbor TCP
connections.

This design is aimed at independent researchers, small labs, and fixed two-to-
four-node systems where the capital cost of a switched fabric is hard to justify.

## Project status

The plugin is operational research infrastructure and is used for real
four-node tensor-parallel inference.

Working today:

- NCCL external network-plugin ABI v8 and v9
- Multiple direct-attached RoCE links on distinct IPv4 subnets
- Multi-address handle exchange
- Subnet-aware RDMA NIC selection
- Per-connection RDMA queue pairs
- Typed NCCL opaque objects and persistent request completion state
- Asynchronous connection setup and connection pooling
- Operation timeout and fatal-error propagation
- Forced all-TCP mode with `NCCL_MESH_DISABLE_RDMA=1`
- Opt-in hybrid RDMA/TCP mode with `NCCL_MESH_HYBRID_TCP=1`
- Per-connection operation, byte, and error counters
- Close-time connection summaries
- Four-node NCCL collectives and production vLLM TP=4 inference

Important boundaries:

- Hybrid TCP is intended for sparse non-neighbor setup/control connections, not
  as a line-rate replacement for RDMA.
- There is no active store-and-forward RDMA relay.
- The plugin does not stripe one logical peer connection across multiple QPs.
- Host-pointer support is advertised; GPUDirect RDMA is not advertised.
- Periodic latency metrics are currently unreliable; byte, operation, error,
  and close-summary counters are the useful diagnostics.

## Validated four-node configuration

The August 2026 validation used:

- 4 × NVIDIA DGX Spark systems with one GB10 GPU/process per node
- Linux/AArch64
- CUDA 13.0
- NCCL 2.29.7 for host collective regression tests
- NCCL 2.30.4+cuda13.2 for production-container validation
- vLLM `0.26.1rc1.dev229+g124154a88`
- Two direct-attached CX-7 RoCE links per node
- Separate 10 GbE management network
- Physical/rank order: `spark-a → spark-b → spark-c → spark-d → spark-a`
- `NCCL_ALGO=Ring` for the bulk collective path
- Hybrid TCP enabled for non-neighbor connections

The validated source history is:

```text
2f7371b  Fix NCCL ABI and enable system RDMA transport
89ad5a8  gate1: add typed transport objects and persistent completions
07336ca  gate1.5: add opt-in hybrid TCP fallback
```

The exact deployed `libnccl-net.so` SHA256 was:

```text
717b613b8557f629c0a77161559fd6acd1e9a36b0f5b4dac36458ac64b5ec4c1
```

See [the full validation report](docs/validation/dgx-spark-4node-2026-08.md).

## Measured results

### NCCL collectives

A four-node, 1,000 MiB host benchmark under NCCL 2.29.7 reported:

| Collective | Reported throughput |
|---|---:|
| All-reduce | 13.56 GB/s |
| Broadcast | 13.60 GB/s |
| All-gather | 11.83 GB/s |
| Reduce-scatter | 13.19 GB/s |

A forced non-neighbor pair under the exact production NCCL 2.30.4 library
selected hybrid TCP and moved 100 MiB at approximately 1.14 GB/s with zero
reported connection errors.

A four-node NCCL 2.30.4 benchmark completed successfully. Bulk traffic remained
on direct RDMA; diagonal TCP connections carried only setup/control-scale data.

### Production inference

The exact four-node vLLM TP=4 deployment completed communicator initialization,
model loading, engine startup, HTTP serving, and inference.

A 27-run llama-benchy matrix used 2,048-token prompts, 128 generated tokens,
context depths 0/4K/16K, and concurrency 1/2/4:

| Context depth | Concurrency | Prefill tok/s | Aggregate decode tok/s | TTFR ms |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1774.6 | 35.9 | 1155.8 |
| 0 | 2 | 1589.7 | 42.2 | 1899.3 |
| 0 | 4 | 1938.6 | 54.3 | 3159.7 |
| 4096 | 1 | 1552.8 | 33.1 | 1319.7 |
| 4096 | 2 | 1556.1 | 48.1 | 2245.4 |
| 4096 | 4 | 1660.3 | 50.5 | 3722.7 |
| 16384 | 1 | 1496.2 | 31.6 | 1369.9 |
| 16384 | 2 | 1505.2 | 40.2 | 2037.7 |
| 16384 | 4 | 1600.1 | 47.0 | 3781.6 |

These are validation snapshots, not universal performance guarantees.

## Requirements

Build requirements:

- Linux
- GCC or a compatible C compiler
- GNU Make
- `pkg-config`
- system `libibverbs` headers and library
- POSIX threads and `libdl`

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libibverbs-dev
```

PyTorch and NCCL are needed for the Python integration benchmarks but are not
linked into the plugin.

## Build and test

```bash
git clone https://github.com/autoscriptlabs/nccl-mesh-plugin.git
cd nccl-mesh-plugin

make test-deps
make -j"$(nproc)"
make test
```

The build creates:

```text
libnccl-net.so
libnccl-net-mesh.so -> libnccl-net.so
```

Confirm that the system verbs library is used:

```bash
ldd libnccl-net.so | grep libibverbs
```

Optional installation:

```bash
sudo make install
```

Use `PREFIX=/custom/prefix` to change the install root.

## Network setup

Give each direct cable a subnet shared only by its two endpoints:

```text
spark-a ↔ spark-b    192.168.100.0/30
spark-b ↔ spark-c    192.168.101.0/30
spark-c ↔ spark-d    192.168.102.0/30
spark-d ↔ spark-a    192.168.103.0/30
```

Every node must also share a management/control network.

Inspect the local state with:

```bash
ip -br -4 address
ibv_devices
ibv_devinfo
show_gids
```

See [docs/SETUP.md](docs/SETUP.md) for the detailed checklist.

## Run with NCCL

The plugin can be selected by full path:

```bash
export NCCL_NET_PLUGIN=/path/to/libnccl-net.so
export NCCL_NET=Mesh
```

Or, when the plugin directory is on `LD_LIBRARY_PATH`, by suffix:

```bash
export LD_LIBRARY_PATH=/path/to/nccl-mesh-plugin:${LD_LIBRARY_PATH:-}
export NCCL_NET_PLUGIN=mesh
export NCCL_NET=Mesh
```

Pin NCCL bootstrap and hybrid TCP to the shared management interface:

```bash
export NCCL_SOCKET_IFNAME='=enP7s7'
export NCCL_SOCKET_FAMILY=AF_INET
```

Replace `enP7s7` with the interface that can reach every host.

Enable per-connection hybrid fallback:

```bash
export NCCL_MESH_HYBRID_TCP=1
export NCCL_ALGO=Ring
```

`NCCL_ALGO=Ring` keeps the bulk collective graph aligned with the physical ring.
NCCL can still construct tree or shortcut transport connections during
communicator initialization; hybrid mode prevents those non-neighbor edges from
failing outright.

For long, uneven production startup such as multi-rank model loading, the tested
profile was:

```bash
export NCCL_MESH_FAST_FAIL=0
export NCCL_MESH_TIMEOUT_SEC=900
export NCCL_MESH_FATAL_ON_TIMEOUT=1
export NCCL_MESH_METRICS=0
```

The first failed production attempt used both aggressive QP fast-fail settings
and the default 30-second operation timeout. Both settings were changed for the
successful run, so the validation does not isolate which change was decisive.

A successful initialization includes:

```text
NET/Plugin: Loaded net plugin Mesh (v9)
Using network Mesh
```

## Configuration

### Core NCCL settings

| Variable | Typical value | Purpose |
|---|---|---|
| `NCCL_NET_PLUGIN` | full `.so` path or `mesh` | Loads the external plugin |
| `NCCL_NET` | `Mesh` | Selects the network name exported by the plugin |
| `NCCL_SOCKET_IFNAME` | `'=management0'` | NCCL bootstrap and hybrid TCP interface |
| `NCCL_SOCKET_FAMILY` | `AF_INET` | Restricts bootstrap sockets to IPv4 |
| `NCCL_ALGO` | `Ring` | Aligns bulk traffic with a physical ring |
| `NCCL_DEBUG` | `WARN` or `INFO` | NCCL diagnostics |
| `NCCL_DEBUG_SUBSYS` | `INIT,NET` | Focused startup/transport logs |

### Plugin settings

| Variable | Default | Purpose |
|---|---:|---|
| `NCCL_MESH_GID_INDEX` | `3` | Preferred/fallback RoCE GID index |
| `NCCL_MESH_DEBUG` | `0` | Plugin debug level: 0 off, 1 info, 2 verbose |
| `NCCL_MESH_HYBRID_TCP` | `0` | Use management TCP when no RDMA subnet matches |
| `NCCL_MESH_DISABLE_RDMA` | `0` | Force every plugin connection to TCP |
| `NCCL_MESH_FAST_FAIL` | `0` | Use shorter RDMA QP timeout/retry settings |
| `NCCL_MESH_TIMEOUT_MS` | `5000` | Connection-operation timeout in milliseconds |
| `NCCL_MESH_RETRY_COUNT` | `3` | Connection retry count |
| `NCCL_MESH_CONN_POOL` | `1` | Enable connection pooling |
| `NCCL_MESH_ASYNC_CONNECT` | `1` | Enable asynchronous connection setup |
| `NCCL_MESH_TIMEOUT_SEC` | `30` | Per-operation completion timeout |
| `NCCL_MESH_CONNECT_TIMEOUT_SEC` | `10` | TCP connect/handshake timeout |
| `NCCL_MESH_ACCEPT_TIMEOUT_SEC` | `30` | Accept-queue wait timeout |
| `NCCL_MESH_HEALTH_CHECK_INTERVAL_MS` | `1000` | QP health polling interval |
| `NCCL_MESH_FATAL_ON_TIMEOUT` | `1` | Return a fatal plugin error on timeout |
| `NCCL_MESH_METRICS` | `1` | Enable periodic metrics logs |
| `NCCL_MESH_METRICS_INTERVAL_SEC` | `10` | Periodic metrics interval |

## Testing

C and topology tests:

```bash
make test-unit
make test-integration
```

Focused pair test:

```bash
python3 tests/manual/quick_pair_test.py \
  --rank 0 \
  --world-size 2 \
  --master-ip 10.0.0.170 \
  --size-mb 100
```

Run the corresponding rank-1 command on the peer.

Four-node incremental collective test:

```bash
torchrun \
  --nnodes=4 \
  --nproc-per-node=1 \
  --node-rank="$NODE_RANK" \
  --master-addr=10.0.0.170 \
  --master-port=29500 \
  tests/benchmark_incremental.py
```

## Troubleshooting

### Plugin did not load

Expected:

```text
NET/Plugin: Loaded net plugin Mesh (v9)
```

Checks:

```bash
ls -l /path/to/libnccl-net.so
ldd /path/to/libnccl-net.so
echo "$NCCL_NET_PLUGIN"
echo "$NCCL_NET"
```

### Non-neighbor connection fails

Enable hybrid mode and verify that `NCCL_SOCKET_IFNAME` names the shared
management interface:

```bash
export NCCL_MESH_HYBRID_TCP=1
export NCCL_SOCKET_IFNAME='=enP7s7'
```

With `NCCL_MESH_DEBUG=1`, the plugin should log that it selected hybrid TCP for
the non-neighbor peer.

### NCCL progress thread reports `-> 3`

In NCCL, result code 3 is `ncclInternalError`. The production failure that
motivated the tested startup profile occurred after a network-plugin `test()`
call returned that error. Search plugin logs for a completion timeout or RDMA
work-completion failure, then review `NCCL_MESH_TIMEOUT_SEC`,
`NCCL_MESH_FAST_FAIL`, and the reported WC/vendor status.

### Periodic latency values are enormous

The current periodic latency accumulator can underflow or mix timing domains.
Do not use those latency values for performance analysis. Prefer:

- aggregate throughput from the workload benchmark;
- operation/byte/error counters;
- close-time connection summaries.

Disable periodic metrics with:

```bash
export NCCL_MESH_METRICS=0
```

## Known limitations

- Linux verbs/RoCE is the validated platform.
- Hybrid non-neighbor traffic uses the management network.
- No active store-and-forward relay or transparent multi-hop RDMA.
- No single-peer multi-QP striping or physical-lane aggregation.
- Host-pointer support is advertised; GPUDirect RDMA is not.
- Periodic latency metrics need repair.
- The hand-maintained NCCL ABI header must be reviewed when the external plugin
  interface changes.
- Compatibility is validated for the versions listed above, not every NCCL or
  vendor RDMA release.

## Documentation

- [Quick start](QUICKSTART.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Hardware and network setup](docs/SETUP.md)
- [August 2026 four-node validation](docs/validation/dgx-spark-4node-2026-08.md)
- [sparkrun integration example](examples/sparkrun/README.md)

## Contributing

Bug reports should include:

- plugin commit and binary SHA256;
- NCCL, CUDA, and runtime versions;
- GPU and RDMA hardware;
- physical topology and rank order;
- relevant `ip -br -4 address`, `ibv_devices`, and GID information;
- focused `NCCL_DEBUG=INFO` transport logs;
- whether hybrid mode and `NCCL_ALGO=Ring` were enabled.

Do not post credentials, private keys, or private infrastructure details.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

Developed to make direct-attached RDMA clusters practical for independent
researchers and small labs.

## Commercial support

Integration and debugging engagements are available at standard consulting
rates: `autoscriptlabs@gmail.com`.

Services are subject to applicable U.S. export-control restrictions.

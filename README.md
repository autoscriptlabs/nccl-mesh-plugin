# NCCL Mesh Plugin

**A practical NCCL network plugin for direct-attached RoCE clusters that do not justify the cost of a switched RDMA fabric.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Operational](https://img.shields.io/badge/status-operational-brightgreen.svg)](#project-status)

## Overview

NCCL Mesh makes direct-attached RDMA topologies usable as a serious alternative
to a conventional switched fabric for small GPU clusters.

Instead of requiring every node to share one switched InfiniBand or RoCE fabric,
each node can use multiple point-to-point RoCE links, with each cable on its own
IPv4 subnet. The plugin advertises all usable local RDMA addresses and selects
the local NIC whose subnet matches the peer address for a connection.

This design trades some fabric generality for dramatically lower infrastructure
cost. For independent researchers, small labs, and self-funded teams, that can
be the difference between owning a useful multi-node GPU system and being priced
out by a switch that costs as much as, or more than, the compute nodes it connects.

The current implementation has been validated on a four-node NVIDIA DGX Spark
cluster arranged as a physical ring:

```text
spark-a ───── spark-b
   │             │
   │             │
spark-d ───── spark-c
```

Each node has one direct RoCE link to each neighbor and a separate 10 GbE
management network for rendezvous and NCCL out-of-band traffic.

This cluster is used for real multi-node workloads, including frontier-model
inference with tensor parallelism across all four nodes (`TP=4`). The project
should therefore be understood as operational research infrastructure with
known topology constraints, not as a toy transport or a synthetic topology demo.

## Why use this instead of a switch?

A switched fabric remains the most flexible option. It supports arbitrary peer
communication patterns and lets NCCL choose from a wider range of collective
graphs without regard to physical adjacency.

But for a small fixed cluster, that flexibility can be economically
disproportionate. A direct-attached ring uses the RDMA ports already present on
the nodes and replaces an expensive central switch with a small number of
point-to-point cables.

The practical trade is:

| Direct-attached NCCL Mesh | Switched RDMA fabric |
|---|---|
| Much lower capital cost | Higher capital cost |
| No central fabric switch | Arbitrary node-to-node reachability |
| Excellent fit for fixed 2–4 node systems | Better fit for larger or changing clusters |
| Physical rank order matters | Topology is largely abstracted |
| Ring-compatible NCCL graphs currently required | Tree, ring, and other graphs can be selected freely |
| Requires deliberate cabling and subnet design | Requires fabric configuration and switch management |

NCCL Mesh is not intended to pretend that these systems are identical. It is
intended to make the direct-attached design a valid, documented, and performant
choice where the economics of a switched fabric do not make sense.

## Project status

This project is **operational and actively developed**. It has completed sustained
four-node NCCL collective tests and is used for real distributed inference
workloads. It is still narrower in scope than a general switched-fabric
transport, and its current constraints should be treated as part of the system
design rather than hidden implementation details.

What is working today:

- NCCL network-plugin ABI v8 and v9 support
- Direct-attached RoCE links on separate IPv4 subnets
- Multi-address handle exchange
- Subnet-aware NIC selection
- Per-connection RDMA queue pairs
- TCP-based connection handshakes
- Connection pooling and asynchronous connection setup
- Operation timeouts, fatal-error propagation, and periodic metrics
- TCP fallback when RDMA initialization is unavailable or explicitly disabled
- Two-node pair collectives
- Four-node ring collectives when NCCL's logical ring matches the physical ring
- CUDA 13-compatible `cudaPointerAttributes` handling

Important current boundaries:

- The active connection path still requires a directly reachable peer address on
  a local subnet.
- Transparent relay of traffic to non-adjacent nodes is **not wired into the
  active `connect()` path**, despite routing and relay scaffolding elsewhere in
  the source tree.
- Sparse physical rings should currently run with `NCCL_ALGO=Ring`. Tree
  algorithms may attempt direct communication between ranks that are not
  physically adjacent.
- The plugin exposes multiple NICs to NCCL, but does not yet stripe one logical
  peer connection across multiple QPs or aggregate multiple physical lanes into
  one faster edge.
- The plugin currently advertises host-pointer support. GPUDirect RDMA is not
  exposed as a supported plugin capability.

## Why this exists

A direct-cabled cluster differs from a conventional RDMA fabric:

```text
Switched fabric

node-a ─┐
node-b ─┼── RDMA switch ── shared fabric/subnet
node-c ─┤
node-d ─┘
```

```text
Direct-attached ring

node-a:link-ab ───── node-b:link-ba
node-b:link-bc ───── node-c:link-cb
node-c:link-cd ───── node-d:link-dc
node-d:link-da ───── node-a:link-ad
```

In the second layout, each cable can have a different subnet. A peer may advertise
several addresses, but only one of the local interfaces can reach a given
address. NCCL Mesh handles that address-to-NIC selection inside the plugin.

## Validated configuration

The August 2026 validation run used:

- 4 × NVIDIA DGX Spark systems with GB10 GPUs
- Linux/AArch64
- CUDA 13.0
- NCCL 2.29.7
- PyTorch 2.11 development build
- Two direct-attached RoCE interfaces per node
- Each RDMA interface reporting 200,000 Mb/s link speed
- Separate 10 GbE management network
- One process and one GPU per node
- Physical and logical rank order: `spark-a → spark-b → spark-c → spark-d → spark-a`
- `NCCL_ALGO=Ring`

A successful initialization includes messages similar to:

```text
NET/Plugin: Loaded net plugin Mesh (v9)
Using network Mesh
Connected all rings
Channel ... via NET/Mesh/0
Channel ... via NET/Mesh/1
```

## Measured results

One four-node run of `tests/benchmark_incremental.py` produced the following
1,000 MiB results:

| Collective | Average time | Reported throughput |
|---|---:|---:|
| All-reduce | 116.30 ms | 13.52 GB/s ring bus bandwidth |
| Broadcast | 77.53 ms | 13.52 GB/s payload throughput |
| All-gather | 266.94 ms | 11.78 GB/s aggregate remote bytes per rank |
| Reduce-scatter | 59.49 ms | 13.22 GB/s per-rank moved bytes |

These metrics use different collective-specific normalizations and should not be
compared as though they were the same quantity. For all-reduce, the benchmark
also records algorithm bandwidth and applies the conventional ring bus factor:

```text
bus_bw = algorithm_bw × 2 × (nranks - 1) / nranks
```

The measurements are a validation snapshot, not a line-rate claim or a general
performance guarantee.

## Requirements

### Build requirements

- Linux
- GCC or a compatible C compiler
- GNU Make
- `pkg-config`
- `libibverbs` development headers and library
- POSIX threads and `libdl`

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libibverbs-dev
```

PyTorch and NCCL are required to run the Python benchmarks, but are not linked
into the plugin at build time.

### Runtime requirements

- RoCE-capable RDMA interfaces
- IPv4 addresses assigned to the direct links
- A compatible NCCL release with external network-plugin support
- The same plugin build available on every participating node
- A management or control network reachable by every node

## Build

```bash
git clone https://github.com/autoscriptlabs/nccl-mesh-plugin.git
cd nccl-mesh-plugin

make test-deps
make -j"$(nproc)"
```

The build produces:

```text
libnccl-net.so
libnccl-net-mesh.so -> libnccl-net.so
```

Confirm that the result uses the system verbs library:

```bash
ldd libnccl-net.so | grep libibverbs
```

Optional installation:

```bash
sudo make install
```

Use `PREFIX=/custom/prefix` to change the install root.

## Configure the network

Each direct link should have addresses in a subnet shared only by the two
endpoints of that cable. For example:

```text
spark-a ↔ spark-b    192.168.100.0/30
spark-b ↔ spark-c    192.168.101.0/30
spark-c ↔ spark-d    192.168.102.0/30
spark-d ↔ spark-a    192.168.103.0/30
```

The exact address plan is site-specific. What matters is that both endpoints of
each cable agree on the subnet and that unrelated links do not accidentally
overlap.

Check the local configuration with:

```bash
ip -br -4 address
ibv_devices
ibv_devinfo
```

For RoCE, also inspect the GID table:

```bash
show_gids
# or
cat /sys/class/infiniband/*/ports/1/gids/*
```

## Run with NCCL

Place the plugin directory on the runtime library path and select the `mesh`
plugin suffix:

```bash
export LD_LIBRARY_PATH=/path/to/nccl-mesh-plugin:${LD_LIBRARY_PATH:-}
export NCCL_NET_PLUGIN=mesh
export NCCL_NET=Mesh
```

NCCL resolves `NCCL_NET_PLUGIN=mesh` as `libnccl-net-mesh.so`. `NCCL_NET=Mesh`
forces the communicator to use the network name exported by this plugin instead
of silently selecting another transport.

Use the shared management interface for NCCL socket bootstrap and out-of-band
control traffic:

```bash
export NCCL_SOCKET_IFNAME='=enP7s7'
export NCCL_SOCKET_FAMILY=AF_INET
```

Replace `enP7s7` with the management-interface name on the local host. This
setting does **not** move collective payload traffic onto the management NIC;
the Mesh plugin still selects direct RDMA interfaces by peer subnet.

For a sparse four-node physical ring:

```bash
export NCCL_ALGO=Ring
```

Keep the NCCL rank order aligned with the physical cable order.

### Four-node benchmark

Run one process per node. On each node, assign a unique node rank from 0 through
3:

```bash
torchrun \
  --nnodes=4 \
  --nproc-per-node=1 \
  --node-rank="$NODE_RANK" \
  --master-addr=10.0.0.170 \
  --master-port=29500 \
  tests/benchmark_incremental.py
```

The benchmark writes one JSON object per line to:

```text
benchmark_results_YYYYMMDD_HHMMSS.jsonl
```

Rank 0 writes results after every completed test so useful data survives a later
failure.

### Two-node manual test

```bash
python3 tests/manual/quick_pair_test.py \
  --rank 0 \
  --world-size 2 \
  --master-ip 10.0.0.170 \
  --size-mb 1000
```

Run the corresponding rank-1 command on the peer.

## How the plugin works

### 1. NIC discovery

The plugin enumerates verbs devices with `ibv_get_device_list()`, opens each
usable device, maps it to its Linux network interface, reads the interface
address and netmask, and records link properties.

### 2. Multi-address listen handles

`listen()` creates resources across the available NICs and returns a compact NCCL
handle containing the addresses a peer may use.

### 3. Subnet-aware connection selection

`connect()` examines the peer's advertised addresses. It first prefers a
high-speed local NIC on the same subnet, then considers any matching local NIC.
If no advertised address shares a subnet with a local interface, the connection
fails with `ncclSystemError`.

### 4. Queue-pair handshake

A TCP handshake exchanges QP information, GIDs, NIC indexes, and related
connection metadata. The selected RDMA QPs are then transitioned to the connected
state.

### 5. Data transfer and completion

The plugin implements the NCCL network-plugin send, receive, flush, registration,
completion, and close operations using verbs resources. It also monitors timeout
and fatal-error conditions to avoid indefinite silent hangs.

## Configuration

### NCCL settings commonly used with this plugin

| Variable | Typical value | Purpose |
|---|---|---|
| `NCCL_NET_PLUGIN` | `mesh` | Loads `libnccl-net-mesh.so` |
| `NCCL_NET` | `Mesh` | Forces the Mesh network implementation |
| `NCCL_SOCKET_IFNAME` | `'=management0'` | Selects NCCL bootstrap/OOB interface |
| `NCCL_SOCKET_FAMILY` | `AF_INET` | Restricts socket bootstrap to IPv4 |
| `NCCL_ALGO` | `Ring` | Keeps sparse-ring collectives on direct-neighbor edges |
| `NCCL_DEBUG` | `WARN` or `INFO` | Controls NCCL diagnostic output |
| `NCCL_DEBUG_SUBSYS` | `INIT,NET,GRAPH` | Narrows verbose NCCL logging |

### Plugin settings implemented in the current source

| Variable | Default | Purpose |
|---|---:|---|
| `NCCL_MESH_GID_INDEX` | `3` | Preferred/default RoCE GID index |
| `NCCL_MESH_DEBUG` | `0` | Plugin debug level: 0 off, 1 info, 2 verbose |
| `NCCL_MESH_FAST_FAIL` | `0` | Enables faster failure detection paths |
| `NCCL_MESH_TIMEOUT_MS` | `5000` | Connection-operation timeout in milliseconds |
| `NCCL_MESH_RETRY_COUNT` | `3` | Connection retry count |
| `NCCL_MESH_DISABLE_RDMA` | `0` | Forces TCP fallback when set to 1 |
| `NCCL_MESH_CONN_POOL` | `1` | Enables connection pooling |
| `NCCL_MESH_ASYNC_CONNECT` | `1` | Enables asynchronous connection setup |
| `NCCL_MESH_TIMEOUT_SEC` | `30` | Per-operation completion timeout |
| `NCCL_MESH_CONNECT_TIMEOUT_SEC` | `10` | TCP handshake timeout |
| `NCCL_MESH_ACCEPT_TIMEOUT_SEC` | `30` | Accept-queue wait timeout |
| `NCCL_MESH_HEALTH_CHECK_INTERVAL_MS` | `1000` | QP health polling interval |
| `NCCL_MESH_FATAL_ON_TIMEOUT` | `1` | Promotes timeouts to fatal plugin errors |
| `NCCL_MESH_METRICS` | `1` | Enables periodic metrics logging |
| `NCCL_MESH_METRICS_INTERVAL_SEC` | `10` | Metrics reporting interval |

Leave advanced settings at their defaults unless logs point to a specific
problem.

## Testing

Build and run C tests:

```bash
make test-unit
```

Run the Python topology tests:

```bash
make test-integration
```

Run both:

```bash
make test
```

Syntax-check the distributed benchmarks:

```bash
python3 -m py_compile \
  tests/benchmark_incremental.py \
  tests/manual/quick_pair_test.py
```

For transport validation, the Python collective benchmarks are more meaningful
than the topology simulations because they exercise the actual NCCL plugin and
RDMA data path.

## Troubleshooting

### NCCL did not load the plugin

Expected log:

```text
NET/Plugin: Loaded net plugin Mesh (v9)
```

Checks:

```bash
ls -l /path/to/nccl-mesh-plugin/libnccl-net-mesh.so
ldd /path/to/nccl-mesh-plugin/libnccl-net.so
echo "$LD_LIBRARY_PATH"
```

Set:

```bash
export LD_LIBRARY_PATH=/path/to/nccl-mesh-plugin:${LD_LIBRARY_PATH:-}
export NCCL_NET_PLUGIN=mesh
export NCCL_NET=Mesh
```

### `No local NIC found on same subnet as any peer address`

This means the active direct connection path could not find a local interface
that shares a subnet with any address advertised by the peer.

For a physical ring, check:

- Rank order matches cable order.
- `NCCL_ALGO=Ring` is set.
- Both ends of every direct cable use the same subnet and prefix length.
- The expected RDMA interfaces are up.
- The peer advertised the intended link address.
- NCCL did not select a tree or another graph containing a non-neighbor edge.

### Job hangs before the first benchmark result

The first benchmark barrier is itself an NCCL collective. Enable:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
```

Verify that all ranks report communicator initialization, ring connections, and
the expected Mesh NIC assignments.

### NCCL uses the wrong bootstrap interface

Pin NCCL's socket control plane to the shared management NIC:

```bash
export NCCL_SOCKET_IFNAME='=enP7s7'
export NCCL_SOCKET_FAMILY=AF_INET
```

Do not pin `NCCL_SOCKET_IFNAME` to a point-to-point RoCE interface that cannot
reach every node.

### GID-table warnings

Messages such as:

```text
NET/IB: <device>:1 GID table changed
```

usually indicate that an address or GID entry changed while NCCL was running.
Check for network-manager activity, interface resets, duplicate address
configuration, or scripts that reapply link addresses during a job.

### PyTorch reports an unknown compute capability

A warning about an unknown GPU compute capability comes from the PyTorch build,
not from this network plugin. Use a PyTorch build that explicitly supports the
installed GPU architecture.

## Known limitations

- Linux and verbs/RoCE only in the validated configuration
- Direct-subnet peer connections only in the active data path
- No transparent forwarding for non-adjacent peers
- Sparse-ring deployments currently require ring-compatible NCCL graphs
- No single-peer multi-QP striping or physical-lane aggregation
- Host-pointer capability is advertised; GPUDirect RDMA is not
- No claim of compatibility with every NCCL release or vendor RDMA stack
- TCP fallback is functional but is not expected to match RDMA performance
- The hand-maintained NCCL ABI compatibility header must be reviewed when NCCL
  changes its external network-plugin interface

## Roadmap

- Wire topology routing and relay forwarding into the active connection path, or
  remove the unused scaffolding
- Add multi-QP striping and explicit dual-link aggregation
- Add GPUDirect RDMA capability where the platform and memory model permit it
- Test against upstream `nccl-tests`
- Add automated ABI checks against supported NCCL headers
- Add CI builds for AArch64 and x86_64
- Replace host-specific benchmark launch steps with reusable cluster tooling
- Expand failure-injection and link-flap testing

## Repository layout

```text
src/
  mesh_plugin.c              Core NCCL plugin and RDMA transport
  mesh_routing.c             Topology, routing, and relay scaffolding

include/
  mesh_plugin.h              Core plugin data structures
  mesh_routing.h             Routing and relay structures
  nccl/net.h                 Local NCCL network-plugin ABI compatibility header

tests/
  benchmark_incremental.py   Four-collective incremental benchmark
  manual/quick_pair_test.py  Focused two-rank all-reduce test
  test_routing.c             Routing unit tests
  test_error_paths.c         Error-path tests
  test_ring_topo.py          Ring-topology simulation
  test_line_topo.py          Line-topology simulation

docs/
  ARCHITECTURE.md
  SETUP.md
  PARTIAL_MESH_ROUTING_PLAN.md
```

## Contributing

Bug reports should include:

- NCCL and CUDA versions
- GPU and RDMA hardware
- `ibv_devices` and relevant `ip -br -4 address` output
- Physical topology and rank order
- Plugin build commit
- Relevant `NCCL_DEBUG=INFO` logs
- Whether the failure reproduces with `NCCL_ALGO=Ring`

Avoid posting private hostnames, credentials, or routable infrastructure
addresses.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

Developed to explore direct-attached RDMA topologies for small GPU clusters,
including NVIDIA DGX Spark systems.

## Commercial support

Integration and debugging engagements are available at standard consulting
rates: `autoscriptlabs@gmail.com`.

Services are subject to applicable U.S. export-control restrictions.

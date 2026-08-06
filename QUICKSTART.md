# Quick Start

This guide builds the plugin, checks a direct-attached RoCE ring, and runs NCCL
with optional hybrid TCP for non-neighbor connections.

## 1. Build

On every architecture that will load the plugin:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libibverbs-dev

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

Confirm the verbs dependency:

```bash
ldd libnccl-net.so | grep libibverbs
sha256sum libnccl-net.so
```

Deploy the **same binary** to every node.

## 2. Check the topology

Every direct cable needs a subnet shared only by its endpoints. A four-node ring
can use:

```text
spark-a ↔ spark-b    192.168.100.0/30
spark-b ↔ spark-c    192.168.101.0/30
spark-c ↔ spark-d    192.168.102.0/30
spark-d ↔ spark-a    192.168.103.0/30
```

Every host must also share one management network.

On each node:

```bash
ip -br -4 address
ibv_devices
ibv_devinfo
show_gids
```

Verify direct neighbors with `ping` and an RDMA bandwidth tool before testing
NCCL.

## 3. Select the plugin

Using a full path:

```bash
export NCCL_NET_PLUGIN=/absolute/path/to/libnccl-net.so
export NCCL_NET=Mesh
```

Or using the installed/symlinked suffix:

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

## 4. Choose the transport policy

Direct RDMA only:

```bash
unset NCCL_MESH_HYBRID_TCP
unset NCCL_MESH_DISABLE_RDMA
```

Hybrid per connection:

```bash
export NCCL_MESH_HYBRID_TCP=1
export NCCL_ALGO=Ring
```

All TCP, for diagnosis:

```bash
export NCCL_MESH_DISABLE_RDMA=1
```

Do not set both hybrid mode and forced all-TCP mode unless the all-TCP behavior
is deliberate.

## 5. Production startup profile

For workloads where ranks can spend several minutes in different model-loading
or compilation phases, the August 2026 production validation used:

```bash
export NCCL_MESH_FAST_FAIL=0
export NCCL_MESH_TIMEOUT_SEC=900
export NCCL_MESH_FATAL_ON_TIMEOUT=1
export NCCL_MESH_METRICS=0

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
```

The successful run changed both `NCCL_MESH_FAST_FAIL` and
`NCCL_MESH_TIMEOUT_SEC`, so the validation does not identify one setting as the
sole fix.

## 6. Run a focused pair test

Start rank 0:

```bash
python3 tests/manual/quick_pair_test.py \
  --rank 0 \
  --world-size 2 \
  --master-ip 10.0.0.170 \
  --size-mb 100
```

Run rank 1 on the peer with `--rank 1`.

For a forced non-neighbor test in a four-node ring, choose diagonal hosts and
enable hybrid mode. With `NCCL_MESH_DEBUG=1`, confirm a log equivalent to:

```text
selecting hybrid TCP
```

## 7. Run four-node collectives

One process per host:

```bash
torchrun \
  --nnodes=4 \
  --nproc-per-node=1 \
  --node-rank="$NODE_RANK" \
  --master-addr=10.0.0.170 \
  --master-port=29500 \
  tests/benchmark_incremental.py
```

Keep the rank order aligned with the physical cable order.

## 8. Verify the active path

Expected NCCL messages:

```text
NET/Plugin: Loaded net plugin Mesh (v9)
Using network Mesh
```

Expected plugin behavior:

- neighbor connections report RDMA;
- diagonal connections report hybrid TCP when needed;
- close summaries show nonzero bytes on the transport that actually carried
  data;
- error counters remain zero.

A four-node ring can still create diagonal tree/shortcut connections during
communicator initialization even with `NCCL_ALGO=Ring`. Hybrid mode handles
those connections while bulk ring traffic remains on RDMA.

## sparkrun

See [examples/sparkrun/README.md](examples/sparkrun/README.md) for staging the
plugin into a container-visible cache path and adding the tested environment
block to a recipe.

## Common failures

### HTTP or runtime starts, but NCCL plugin is absent

Check the environment inside the worker/container:

```bash
env | grep -E '^NCCL_(NET|NET_PLUGIN|SOCKET_IFNAME)'
```

Verify that the full plugin path exists inside the container.

### `No local NIC found on same subnet`

For a direct neighbor, inspect address and prefix configuration on both ends.

For a non-neighbor, enable:

```bash
export NCCL_MESH_HYBRID_TCP=1
```

and ensure `NCCL_SOCKET_IFNAME` is the shared management interface.

### `transport/net.cc ... -> 3`

Result code 3 is `ncclInternalError`. Search the plugin log for:

- completion timeout;
- RDMA WC status and vendor error;
- peer failure;
- connection summary with a nonzero error count.

### Periodic latency values look impossible

The periodic latency accumulator is a known diagnostic bug. Disable it with:

```bash
export NCCL_MESH_METRICS=0
```

Use workload throughput and close-time counters instead.

# Hardware and Network Setup

## Reference topology

The validated deployment uses four DGX Spark systems in a physical ring:

```text
spark-a ───── spark-b
   │             │
   │             │
spark-d ───── spark-c
```

Each host has:

- one direct RoCE link to its clockwise neighbor;
- one direct RoCE link to its counter-clockwise neighbor;
- one shared management interface reachable by every host.

The management network is used for orchestration, NCCL bootstrap, and optional
hybrid TCP. Direct neighbors use RoCE for NCCL payloads.

## Address plan

Use one non-overlapping subnet per cable. `/30` is sufficient for a two-endpoint
IPv4 link:

| Link | Example subnet |
|---|---|
| spark-a ↔ spark-b | `192.168.100.0/30` |
| spark-b ↔ spark-c | `192.168.101.0/30` |
| spark-c ↔ spark-d | `192.168.102.0/30` |
| spark-d ↔ spark-a | `192.168.103.0/30` |

The exact addresses and interface names are site-specific. Do not reuse the same
subnet on unrelated cables.

## Identify the devices

```bash
ibv_devices
rdma link show
for d in /sys/class/infiniband/*/device/net/*; do
  readlink -f "$d"
done
ip -br link
ip -br -4 address
```

Record the mapping between:

- Linux interface;
- verbs device;
- physical cable peer;
- IPv4 address/prefix;
- usable RoCEv2 GID index.

## Configure each direct link

Example only:

```bash
sudo ip addr add 192.168.100.1/30 dev <spark-a-to-b>
sudo ip link set <spark-a-to-b> up
```

Configure the peer with the other usable address from the same `/30`.

Make the configuration persistent with the host's network manager only after
the temporary configuration has been verified.

## Verify link-local IP connectivity

From each host, ping only its two physical neighbors on the corresponding direct
addresses:

```bash
ping -c 3 <neighbor-direct-ip>
```

A non-neighbor direct address is not expected to be reachable in a sparse ring.

Verify that every host can reach every other host on the management network.

## Verify RoCE

Inspect GIDs:

```bash
show_gids
# or
for f in /sys/class/infiniband/*/ports/1/gids/*; do
  printf '%s: ' "$f"
  cat "$f"
done
```

Run an RDMA bandwidth test on every cable in both directions. Use the verbs
device and GID index that correspond to that link.

Do not proceed to NCCL until the point-to-point RDMA tools are stable.

## Build and distribute the plugin

```bash
make clean
make test-deps
make -j"$(nproc)"
make test
sha256sum libnccl-net.so
```

Copy the same AArch64 build to every Spark and verify the checksum on all hosts.

Example:

```bash
for host in spark-a spark-b spark-c spark-d; do
  ssh "$host" 'sha256sum ~/.cache/huggingface/nccl-mesh/libnccl-net.so'
done
```

The August 2026 validated binary hash was:

```text
717b613b8557f629c0a77161559fd6acd1e9a36b0f5b4dac36458ac64b5ec4c1
```

A future build will have a different hash; what matters is exact equality across
the participating hosts.

## Runtime environment

Basic plugin selection:

```bash
export NCCL_NET_PLUGIN=/path/visible/to/the/runtime/libnccl-net.so
export NCCL_NET=Mesh
export NCCL_SOCKET_IFNAME='=enP7s7'
export NCCL_SOCKET_FAMILY=AF_INET
```

Sparse ring policy:

```bash
export NCCL_ALGO=Ring
export NCCL_MESH_HYBRID_TCP=1
```

Recommended diagnostics during bring-up:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_MESH_DEBUG=1
```

Production startup profile validated with the four-node vLLM workload:

```bash
export NCCL_MESH_FAST_FAIL=0
export NCCL_MESH_TIMEOUT_SEC=900
export NCCL_MESH_FATAL_ON_TIMEOUT=1
export NCCL_MESH_METRICS=0
```

`NCCL_SOCKET_IFNAME` must name the all-reachable management interface. Do not
pin it to one point-to-point RoCE interface.

## Rank order

Keep the distributed rank order aligned with the cable order:

```text
rank 0: spark-a
rank 1: spark-b
rank 2: spark-c
rank 3: spark-d
```

That makes the primary NCCL ring edges physical neighbors.

Hybrid mode handles non-neighbor connections that NCCL may construct for tree or
shortcut setup. It does not make arbitrary rank orders equally efficient.

## Validation sequence

Use this order so failures stay local and diagnosable:

1. IP ping across each physical cable.
2. Point-to-point RDMA tool across each cable.
3. Plugin unit and topology tests.
4. Adjacent two-rank NCCL transfer with RDMA.
5. Diagonal two-rank transfer with hybrid TCP.
6. Four-node NCCL collective benchmark.
7. Exact production runtime startup.
8. Small inference request.
9. Sustained benchmark matrix.

## Expected transport evidence

Adjacent pair:

```text
transport=RDMA
```

Diagonal pair with hybrid enabled:

```text
selecting hybrid TCP
transport=TCP
```

At connection close, inspect:

- transport kind;
- operation count;
- bytes;
- error count.

In the validated four-node NCCL 2.30.4 benchmark, diagonal TCP connections
carried only setup/control-scale bytes while the bulk collective path stayed on
RDMA.

## Container deployment

A runtime container must see:

- the same plugin binary;
- the selected NCCL library;
- the shared management interface;
- the direct RoCE interfaces;
- a writable location for optional NCCL debug files.

With sparkrun, a practical shared convention is:

```text
host:      ~/.cache/huggingface/nccl-mesh/libnccl-net.so
container: /cache/huggingface/nccl-mesh/libnccl-net.so
```

Verify the path inside the worker container before launching the full workload.

## GID changes during a job

Warnings such as:

```text
NET/IB: <device>:1 GID table changed
```

indicate interface/GID churn. Check for:

- NetworkManager or netplan reapplying addresses;
- interface reset or link flap;
- duplicate address configuration;
- scripts modifying the direct interfaces during the run.

Stable direct-link addressing is required for reliable QP operation.

## Firewall

Allow the NCCL bootstrap and plugin listener traffic on the trusted management
network. The plugin uses dynamically allocated listener ports; a narrow fixed
port range is not currently documented by the implementation.

For initial isolated-lab testing, validate with host firewalls accounted for
rather than guessing a port range.

## Safety notes

- Never copy a personal GitHub token or workstation private SSH key onto every
  cluster node.
- Use a dedicated deploy key, short-lived credential, or stage source/artifacts
  from an authenticated workstation.
- Publish only private-address examples and redacted logs.

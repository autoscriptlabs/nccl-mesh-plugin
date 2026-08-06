# Architecture

## Design goal

NCCL Mesh provides a network-plugin transport for hosts connected by multiple
point-to-point RoCE subnets. It solves two separate reachability cases:

1. peers sharing one direct RDMA subnet;
2. peers that do not share an RDMA subnet but can reach each other on a common
   management network.

The second case is optional and uses direct TCP between the two endpoints. It
does not relay data through an intermediate rank.

## Active transport model

```text
                    peer handle
                        │
                        ▼
             find shared RDMA subnet?
                 │                 │
                yes                no
                 │                 │
                 ▼                 ▼
          direct verbs QP    hybrid enabled?
                                │       │
                               yes      no
                                │       │
                                ▼       ▼
                       management TCP   error
```

The transport choice is made independently for each NCCL connection. A four-node
physical ring can therefore use RDMA for adjacent bulk edges and TCP only for
non-neighbor tree/shortcut connections.

## NCCL object typing

NCCL passes opaque pointers for listeners, send communicators, receive
communicators, memory registrations, and requests. Gate 1 introduced explicit
object kinds and validation so operation dispatch does not infer an object's
meaning from incidental fields.

The active object families are:

- listener;
- RDMA send communicator;
- RDMA receive communicator;
- TCP send communicator;
- TCP receive communicator;
- registered memory handle;
- persistent completion request.

Typed dispatch is important in hybrid mode because the same NCCL API entry point
can receive an RDMA-backed or TCP-backed object.

## Listen handle

The plugin's packed listen handle stays within NCCL's 128-byte network-handle
limit. The Gate 1.5 build validated an actual size of 120 bytes.

The handle advertises:

- direct RDMA address candidates;
- subnet information needed for local NIC selection;
- listener information for connection setup;
- a management TCP endpoint for hybrid mode.

The management endpoint is discovered independently from the direct RDMA
interfaces and follows the interface selected by `NCCL_SOCKET_IFNAME`.

## Listener and connection setup

`listen()` creates a mixed listener capable of accepting:

- an RDMA QP handshake; or
- a hybrid TCP data connection.

Connection setup remains asynchronous so two ranks can initiate connections
without blocking each other in a symmetric connect/accept deadlock.

### Direct RDMA path

For a peer with a shared subnet:

1. inspect every advertised peer RDMA address;
2. find a usable local NIC on the same subnet;
3. create a connection-specific QP and completion resources;
4. exchange QP metadata over the listener handshake;
5. transition the QP to the connected state;
6. return typed RDMA send/receive communicators.

Bulk payload transfer uses verbs send/receive operations and registered host
memory.

### Hybrid TCP path

For a peer with no shared RDMA subnet and `NCCL_MESH_HYBRID_TCP=1`:

1. select the peer's management endpoint;
2. establish a direct TCP stream between the two NCCL endpoints;
3. create typed TCP send/receive communicators;
4. move the NCCL payload over that stream;
5. use the same persistent request API expected by NCCL.

This is an endpoint-to-endpoint fallback. It does not create an intermediate
forwarding process and does not consume another rank as a relay.

### Forced TCP path

`NCCL_MESH_DISABLE_RDMA=1` bypasses RDMA selection and sends every plugin
connection over TCP. This is primarily a diagnostic mode.

## Persistent completion requests

Gate 1 replaced stack-like or transient completion state with requests that
remain valid across repeated NCCL `test()` calls.

A request records:

- object/transport type;
- completion state;
- transferred size;
- start/progress timing;
- error result;
- transport-specific completion context.

`test()` dispatches by request type:

- RDMA requests poll the relevant CQ;
- TCP requests advance nonblocking stream I/O;
- completed requests return the stored result and size.

The request is not considered complete until the transport has either completed
the operation or produced a terminal error.

## Error and timeout behavior

### Per-operation timeout

`NCCL_MESH_TIMEOUT_SEC` defaults to 30 seconds. When an outstanding RDMA
operation produces no completion before that interval, the request is completed
with an error. With `NCCL_MESH_FATAL_ON_TIMEOUT=1` (default), the plugin returns
a fatal NCCL result rather than allowing an indefinite silent wait.

Long model-loading workloads can leave ranks at different initialization stages
for minutes. The validated production profile raised this timeout to 900
seconds.

### RDMA work-completion errors

A non-success verbs work completion records the WC status and vendor error and
returns a terminal NCCL error to the caller.

### TCP errors

TCP connect, read, write, or peer-close failures are propagated through the
request rather than being converted into a successful zero-byte completion.

### Fast-fail mode

`NCCL_MESH_FAST_FAIL=1` shortens QP timeout/retry settings. It does not replace
the plugin's per-operation timer. The production validation used the default
`0` after an earlier startup failure under the aggressive setting.

## Counters and summaries

Each connection tracks:

- operation count;
- bytes transferred;
- terminal error count;
- transport kind.

Close-time summaries are the most reliable plugin-level diagnostics.

Periodic metrics can also be enabled, but the current latency accumulator can
underflow or combine incompatible timing values. Until that bug is repaired,
ignore periodic average/max latency and use workload throughput plus the
operation/byte/error counters.

## Topology and routing scaffolding

`src/mesh_routing.c` contains topology and relay scaffolding, but there is no
active store-and-forward relay in the connection or data path.

The current sparse-ring strategy is:

- choose a physical/rank order that keeps bulk NCCL ring edges adjacent;
- use `NCCL_ALGO=Ring`;
- enable hybrid TCP so any non-neighbor setup/tree/shortcut connection can still
  be established.

The four-node production benchmark showed the diagonal TCP connections carrying
only setup/control-scale traffic while bulk payload remained on RDMA. That result
does not justify adding full RDMA relay complexity yet.

## Memory capability

The plugin advertises host-pointer capability. It does not advertise GPUDirect
RDMA support in the validated build.

## ABI support

The repository contains a local NCCL network-plugin ABI compatibility header and
exports v8 and v9 plugin symbols. This header must be reviewed whenever NCCL
changes its external plugin interface.

Validated runtime versions are listed in
[docs/validation/dgx-spark-4node-2026-08.md](validation/dgx-spark-4node-2026-08.md).

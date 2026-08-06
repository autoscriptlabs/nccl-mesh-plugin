# Changelog

All notable changes to this project are documented here.

The project has not yet declared a stable semantic-versioning policy.

## Unreleased

### Added

- Opt-in `NCCL_MESH_HYBRID_TCP=1` transport selection.
- Per-connection direct TCP fallback when peers do not share an RDMA subnet.
- Mixed listener support for RDMA handshakes and TCP data connections.
- Typed transport objects and typed operation dispatch.
- Persistent completion requests across repeated NCCL `test()` calls.
- Per-connection operation, byte, and error counters.
- Close-time transport summaries.
- Four-node NCCL 2.30.4 and production vLLM TP=4 validation report.
- sparkrun integration example.

### Changed

- Directly reachable peers remain on RDMA in hybrid mode.
- Error and timeout paths now preserve terminal completion state.
- NCCL logger calls use the NET subsystem bit expected by current NCCL logging.

### Known issues

- Periodic latency metrics can underflow or combine incompatible timing values.
- The default 30-second operation timeout can be too short for highly uneven
  multi-rank model startup; the validated production profile uses 900 seconds.
- There is no active store-and-forward RDMA relay.

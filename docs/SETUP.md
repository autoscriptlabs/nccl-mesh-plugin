# Hardware Setup Guide

This guide covers setting up a direct-connect RDMA mesh topology with multiple nodes.

## Overview

Our reference setup uses NVIDIA DGX Spark workstations (Grace Hopper architecture with unified memory) connected via direct RDMA cables. The supported topology options are:

- **3 nodes**: Triangle mesh (fully connected) — each node directly connected to all others
- **4+ nodes ring**: Ring topology — each node connects to 2 neighbors, relay routing for non-adjacent
- **Any nodes line**: Line topology — chain of nodes, relay routing for multi-hop

Each ConnectX-7 port provides **200 Gbps** via dual PCIe 5.0 x4 channels using QSFP56 cables.

**Current production cluster**: 4x DGX Spark in a ring topology with 200Gbps links and automatic relay routing for non-adjacent nodes.

## Hardware Requirements

- 3+ nodes with RDMA-capable NICs (ConnectX-7 recommended)
- QSFP56 direct-attach cables (200Gbps per link)
- For triangle mesh: Each node needs 2 NICs
- For ring topology: Each node needs 2 NICs

## Network Topology

### Triangle Mesh (3 Nodes)

```
        Node A
       /      \
   NIC1        NIC2
     |          |
192.168.101.x  192.168.100.x
     |          |
   NIC1        NIC1
     |          |
   Node B ---- Node C
          NIC2
     192.168.102.x
```

### IP Address Assignment

| Link | Subnet | Node A | Node B | Node C |
|------|--------|--------|--------|--------|
| A↔B | 192.168.101.0/24 | .2 | .3 | - |
| A↔C | 192.168.100.0/24 | .2 | - | .3 |
| B↔C | 192.168.102.0/24 | - | .2 | .3 |

## Network Configuration

### 1. Identify NICs

```bash
# List RDMA devices
ibv_devices

# List network interfaces with RDMA
ls -la /sys/class/infiniband/*/device/net/
```

### 2. Configure IP Addresses

On **Node A** (example):

```bash
# Link to Node B
sudo ip addr add 192.168.101.2/24 dev enp1s0f0np0
sudo ip link set enp1s0f0np0 up

# Link to Node C  
sudo ip addr add 192.168.100.2/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up
```

On **Node B**:

```bash
# Link to Node A
sudo ip addr add 192.168.101.3/24 dev enp1s0f0np0
sudo ip link set enp1s0f0np0 up

# Link to Node C
sudo ip addr add 192.168.102.2/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up
```

On **Node C**:

```bash
# Link to Node A
sudo ip addr add 192.168.100.3/24 dev enp1s0f0np0
sudo ip link set enp1s0f0np0 up

# Link to Node B
sudo ip addr add 192.168.102.3/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up
```

### 3. Make Configuration Persistent

Create netplan config (Ubuntu):

```yaml
# /etc/netplan/99-rdma-mesh.yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      addresses:
        - 192.168.101.2/24  # Adjust per node
    enp1s0f1np1:
      addresses:
        - 192.168.100.2/24  # Adjust per node
```

Apply:
```bash
sudo netplan apply
```

## Verify Connectivity

### 1. Ping Test

From Node A:
```bash
ping 192.168.101.3  # Node B
ping 192.168.100.3  # Node C
```

### 2. RDMA Test

```bash
# On Node B (server)
ib_send_bw -d rocep1s0f0 -x 3

# On Node A (client)
ib_send_bw -d rocep1s0f0 -x 3 192.168.101.3
```

Expected output: ~24 GB/s for 200GbE (dual-channel), ~12 GB/s for 100GbE (single-channel)

### 3. Verify GID Index

```bash
# Show GID table
show_gids

# Find RoCE v2 GID (usually index 3)
ibv_devinfo -v | grep -A5 GID
```

## RoCE Configuration

### Enable RoCE v2

```bash
# Check current mode
cat /sys/class/infiniband/rocep*/ports/1/gid_attrs/types/*

# Enable RoCE v2 (if needed)
echo "RoCE v2" | sudo tee /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/0
```

### Configure ECN (Optional but Recommended)

```bash
# Enable ECN for RoCE
sudo sysctl -w net.ipv4.tcp_ecn=1

# Configure PFC (Priority Flow Control) on switch if applicable
```

## Firewall Configuration

Open ports for NCCL communication:

```bash
# TCP ports for handshake (dynamic, 40000-50000 range)
sudo ufw allow 40000:50000/tcp

# Or disable firewall for mesh interfaces
sudo ufw allow in on enp1s0f0np0
sudo ufw allow in on enp1s0f1np1
```

## Troubleshooting

### No RDMA Devices Found

```bash
# Load kernel modules
sudo modprobe ib_core
sudo modprobe mlx5_core
sudo modprobe mlx5_ib

# Check dmesg
dmesg | grep -i mlx
```

### Link Not Coming Up

```bash
# Check physical connection
ethtool enp1s0f0np0

# Check for errors
ip -s link show enp1s0f0np0
```

### RDMA Connection Fails

```bash
# Verify GID is populated
cat /sys/class/infiniband/rocep1s0f0/ports/1/gids/3

# Check RDMA CM
rdma link show
```

### Wrong GID Index

Try different GID indices:

```bash
export NCCL_MESH_GID_INDEX=0  # or 1, 2, 3...
```

## Scaling to 4+ Nodes: Ring Topology

Full mesh becomes impractical beyond 3 nodes (N nodes requires N-1 NICs each, N*(N-1)/2 total links). For 4+ nodes, we use a **ring topology** with automatic relay routing:

### Ring Topology (4 Nodes)

```
        Node A
       /      \
   NIC1        NIC2
     |          |
192.168.101.x  192.168.100.x
     |          |
   NIC1        NIC1
     |          |
   Node B      Node D
     |          |
   NIC2        NIC2
     |          |
192.168.102.x  192.168.103.x
     |          |
   NIC1        NIC2
     \          /
      \        /
       Node C
```

### Ring IP Address Assignment

| Link | Subnet | Node A | Node B | Node C | Node D |
|------|--------|--------|--------|--------|--------|
| A↔B | 192.168.101.0/24 | .2 | .3 | - | - |
| B↔C | 192.168.102.0/24 | - | .2 | .3 | - |
| C↔D | 192.168.103.0/24 | - | - | .2 | .3 |
| D↔A | 192.168.100.0/24 | .3 | - | - | .2 |

### Ring Advantages

- **Only 2 NICs per node** (vs 3 for full mesh with 4 nodes)
- **Only 4 links total** (vs 6 for full mesh)
- **Simpler cabling**: Each node connects to exactly 2 neighbors
- **Dual-path routing**: Opposite nodes can use either CW or CCW path
- **Load balancing**: Traffic automatically balanced across paths

### Ring Communication Patterns

| Source | Destination | Hops | Path Options |
|--------|-------------|------|--------------|
| A | B | 1 | Direct |
| A | C | 2 | A→B→C (CW) or A→D→C (CCW) |
| A | D | 1 | Direct |
| B | D | 2 | B→A→D (CCW) or B→C→D (CW) |

The plugin automatically:
1. Detects the ring topology
2. Computes both CW and CCW paths
3. Balances load across equal-length paths
4. Forwards traffic through relay nodes

### Ring Configuration

```bash
# Enable load balancing (default: on)
export NCCL_MESH_RING_LOAD_BALANCE=1

# Prefer shorter path always (disable load balancing)
export NCCL_MESH_RING_PREFER_SHORT=1

# Threshold before switching paths (default: 1MB)
export NCCL_MESH_RING_BALANCE_THRESHOLD=1048576
```

> **Note**: Full mesh with 4 nodes would require 3 NICs per node, which isn't possible on DGX Spark (only 2 ConnectX-7 ports per node). Ring topology is the only option for 4-node Spark clusters.

## Line Topology

For scenarios where ring cabling isn't possible, line topology is also supported:

### Line Topology (4 Nodes)

```
Node A ---- Node B ---- Node C ---- Node D
  NIC1      NIC1 NIC2   NIC1 NIC2     NIC1
   |          |    |      |    |        |
  192.168.100.x   192.168.101.x   192.168.102.x
```

### Line IP Address Assignment

| Link | Subnet | Node A | Node B | Node C | Node D |
|------|--------|--------|--------|--------|--------|
| A↔B | 192.168.100.0/24 | .2 | .3 | - | - |
| B↔C | 192.168.101.0/24 | - | .2 | .3 | - |
| C↔D | 192.168.102.0/24 | - | - | .2 | .3 |

### Line Communication Patterns

| Source | Destination | Hops | Path |
|--------|-------------|------|------|
| A | B | 1 | Direct |
| A | C | 2 | A→B→C |
| A | D | 3 | A→B→C→D |
| B | D | 2 | B→C→D |

Endpoints (A and D) have only 1 NIC; middle nodes need 2 NICs.

## Reference: DGX Spark Ring Cluster

Our production configuration (4-node ring, 200Gbps QSFP56 links):

| Hostname | Management IP | Mesh IPs | Ring Neighbors |
|----------|--------------|----------|----------------|
| titanic (A) | 10.0.0.170 | 192.168.100.3, 192.168.101.2 | D ↔ A ↔ B |
| iceberg (B) | 10.0.0.171 | 192.168.101.3, 192.168.102.2 | A ↔ B ↔ C |
| carpathia (C) | 10.0.0.172 | 192.168.102.3, 192.168.103.2 | B ↔ C ↔ D |
| water (D) | 10.0.0.173 | 192.168.103.3, 192.168.100.2 | C ↔ D ↔ A |

Non-adjacent pairs (A↔C, B↔D) communicate via automatic relay routing through their neighbors.

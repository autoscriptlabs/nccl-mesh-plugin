# Hardware Setup Guide

This guide covers setting up a direct-connect RDMA mesh topology with multiple nodes.

## Overview

Our reference setup uses NVIDIA DGX Spark workstations (Grace Hopper architecture with unified memory) connected via direct RDMA cables. The topology options are:

- **3 nodes**: Triangle mesh (fully connected) - current production setup
- **4 nodes**: Ring topology (each node connects to 2 neighbors) - planned expansion

Each ConnectX-7 port supports up to **200 Gbps** via dual PCIe 5.0 x4 channels, though current software uses single-channel mode (100 Gbps per link).

## Hardware Requirements

- 3-4 nodes with RDMA-capable NICs (ConnectX-7 recommended for dual-channel support)
- Direct-attach cables (QSFP56/QSFP112 for 100/200GbE)
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

Expected output: ~12 GB/s for 100GbE

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

## Scaling to 4 Nodes: Ring Topology

Full mesh becomes impractical beyond 3 nodes (N nodes requires N-1 NICs each, N*(N-1)/2 total links). For 4 nodes, we use a **ring topology** instead:

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

### Ring Trade-offs

- **Non-adjacent communication requires relay**: A↔C and B↔D go through intermediate nodes
- **Higher latency for non-neighbors**: 2 hops instead of 1
- **Relay routing required**: Plugin must forward traffic (planned feature)

> **Note**: Full mesh with 4 nodes would require 3 NICs per node, which isn't possible on DGX Spark (only 2 ConnectX-7 ports per node). Ring topology is the only option for 4-node Spark clusters.

## Reference: DGX Spark Mesh

Our tested configuration:

| Hostname | Management IP | Mesh IPs |
|----------|--------------|----------|
| titanic (A) | 10.0.0.170 | 192.168.100.2, 192.168.101.2 |
| iceberg (B) | 10.0.0.171 | 192.168.101.3, 192.168.102.2 |
| carpathia (C) | 10.0.0.172 | 192.168.100.3, 192.168.102.3 |

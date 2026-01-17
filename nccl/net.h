/*
 * NCCL Net Plugin API - main header
 */

#ifndef NCCL_NET_H
#define NCCL_NET_H

#include "err.h"
#include "net_v8.h"
#include "net_v9.h"

// Maximum number of outstanding requests
#define NCCL_NET_MAX_REQUESTS 32

// Use v9 as current version
typedef ncclNet_v9_t ncclNet_t;
typedef ncclNetProperties_v9_t ncclNetProperties_t;

#endif // NCCL_NET_H

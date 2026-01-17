/*
 * NCCL Net Plugin v9 API
 */

#ifndef NCCL_NET_V9_H
#define NCCL_NET_V9_H

#include "err.h"
#include <stddef.h>
#include <stdint.h>

/* Virtual device properties for v9 */
typedef struct {
    int ndevs;
    int* devs;
} ncclNetVDeviceProps_v9_t;

/* Net properties v9 */
typedef struct {
    char* name;
    char* pciPath;
    uint64_t guid;
    int ptrSupport;
    int regIsGlobal;
    int forceFlush;
    int speed;
    int port;
    float latency;
    int maxComms;
    int maxRecvs;
    int netDeviceType;
    int netDeviceVersion;
    ncclNetVDeviceProps_v9_t vProps;
    size_t maxP2pBytes;
    size_t maxCollBytes;
} ncclNetProperties_v9_t;

/* Net device handle for v9 */
typedef void* ncclNetDeviceHandle_v9_t;

/* Net plugin v9 interface */
typedef struct {
    const char* name;
    ncclResult_t (*init)(ncclDebugLogger_t logFunction);
    ncclResult_t (*devices)(int* ndev);
    ncclResult_t (*getProperties)(int dev, ncclNetProperties_v9_t* props);
    ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
    ncclResult_t (*connect)(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_v9_t** sendDevComm);
    ncclResult_t (*accept)(void* listenComm, void** recvComm, ncclNetDeviceHandle_v9_t** recvDevComm);
    ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
    ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
    ncclResult_t (*deregMr)(void* comm, void* mhandle);
    ncclResult_t (*isend)(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request);
    ncclResult_t (*irecv)(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request);
    ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
    ncclResult_t (*test)(void* request, int* done, int* sizes);
    ncclResult_t (*closeSend)(void* sendComm);
    ncclResult_t (*closeRecv)(void* recvComm);
    ncclResult_t (*closeListen)(void* listenComm);
    ncclResult_t (*getDeviceMr)(void* comm, void* mhandle, void** dptr_mhandle);
    ncclResult_t (*irecvConsumed)(void* recvComm, int n, void* request);
    ncclResult_t (*makeVDevice)(int* d, ncclNetVDeviceProps_v9_t* props);
} ncclNet_v9_t;

#endif // NCCL_NET_V9_H

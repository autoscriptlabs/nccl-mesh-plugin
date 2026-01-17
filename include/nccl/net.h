/*
 * Stub nccl/net.h for compilation without NCCL headers
 */
#ifndef NCCL_NET_H
#define NCCL_NET_H

#include <stdint.h>
#include <stddef.h>

/* NCCL result codes */
typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7
} ncclResult_t;

/* NCCL debug log levels */
#define NCCL_LOG_NONE 0
#define NCCL_LOG_VERSION 1
#define NCCL_LOG_WARN 2
#define NCCL_LOG_INFO 3
#define NCCL_LOG_ABORT 4
#define NCCL_LOG_TRACE 5

/* Logger function type */
typedef void (*ncclDebugLogger_t)(int level, unsigned long flags, const char *file,
                                   int line, const char *fmt, ...);

/* Net device handle */
typedef void* ncclNetDeviceHandle_t;

/* Maximum handle size */
#define NCCL_NET_HANDLE_MAXSIZE 128

/* Maximum number of outstanding requests */
#define NCCL_NET_MAX_REQUESTS 32

/* Maximum net size */
#define NCCL_MAX_NET_SIZE_BYTES (1ULL << 31)

/* Device version */
#define NCCL_NET_DEVICE_INVALID_VERSION 0

/* Net device type */
typedef enum {
    NCCL_NET_DEVICE_HOST = 0,
    NCCL_NET_DEVICE_UNPACK = 1
} ncclNetDeviceType;

/* Net properties v8 */
typedef struct {
    char name[256];
    char pciPath[256];
    uint64_t guid;
    int ptrSupport;
    int speed;
    int port;
    int maxComms;
    int maxRecvs;
    int latency;
    int netDeviceType;
    int netDeviceVersion;
} ncclNetProperties_v8_t;

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
    ncclNetDeviceType netDeviceType;
    int netDeviceVersion;
    ncclNetVDeviceProps_v9_t vProps;
    size_t maxP2pBytes;
    size_t maxCollBytes;
} ncclNetProperties_v9_t;

/* Net device handle for v9 */
typedef void* ncclNetDeviceHandle_v9_t;

/* Net plugin v8 interface */
typedef struct {
    const char* name;
    ncclResult_t (*init)(ncclDebugLogger_t logFunction);
    ncclResult_t (*devices)(int* ndev);
    ncclResult_t (*getProperties)(int dev, ncclNetProperties_v8_t* props);
    ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
    ncclResult_t (*connect)(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm);
    ncclResult_t (*accept)(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm);
    ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
    ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
    ncclResult_t (*deregMr)(void* comm, void* mhandle);
    ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
    ncclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request);
    ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
    ncclResult_t (*test)(void* request, int* done, int* sizes);
    ncclResult_t (*closeSend)(void* sendComm);
    ncclResult_t (*closeRecv)(void* recvComm);
    ncclResult_t (*closeListen)(void* listenComm);
    ncclResult_t (*getDeviceMr)(void* comm, void* mhandle, void** dptr_mhandle);
    ncclResult_t (*irecvConsumed)(void* recvComm, int n, void* request);
} ncclNet_v8_t;

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

/* Plugin version */
#define NCCL_NET_PLUGIN_SYMBOL ncclNetPlugin_v9

#endif /* NCCL_NET_H */

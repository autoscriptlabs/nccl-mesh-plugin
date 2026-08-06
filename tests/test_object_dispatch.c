/*
 * Gate 1 typed-object tests.
 *
 * These tests do not require RDMA hardware. They verify the ABI invariant used
 * by all NCCL opaque-pointer dispatch and the persistent request-state prefix.
 */

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "nccl/err.h"
#include "mesh_plugin.h"

static int failures;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                    \
        }                                                                  \
    } while (0)

static void test_layouts(void) {
    CHECK(offsetof(struct mesh_listen_comm, object) == 0);
    CHECK(offsetof(struct mesh_send_comm, object) == 0);
    CHECK(offsetof(struct mesh_recv_comm, object) == 0);
    CHECK(offsetof(struct mesh_tcp_listen_comm, object) == 0);
    CHECK(offsetof(struct mesh_tcp_send_comm, object) == 0);
    CHECK(offsetof(struct mesh_tcp_recv_comm, object) == 0);
    CHECK(offsetof(struct mesh_mr_handle, object) == 0);
    CHECK(offsetof(struct mesh_request, header) == 0);
    CHECK(offsetof(struct mesh_tcp_request, header) == 0);
    CHECK(offsetof(struct mesh_request_header, object) == 0);
}

static void test_object_validation(void) {
    struct mesh_send_comm send;
    memset(&send, 0, sizeof(send));

    CHECK(mesh_object_kind_of(NULL) == MESH_OBJ_INVALID);
    CHECK(mesh_object_kind_of(&send) == MESH_OBJ_INVALID);

    mesh_object_init(&send.object, MESH_OBJ_SEND_RDMA);
    CHECK(mesh_object_kind_of(&send) == MESH_OBJ_SEND_RDMA);
    CHECK(mesh_object_is(&send, MESH_OBJ_SEND_RDMA));
    CHECK(!mesh_object_is(&send, MESH_OBJ_RECV_RDMA));

    send.object.magic ^= 1U;
    CHECK(mesh_object_kind_of(&send) == MESH_OBJ_INVALID);

    mesh_object_init(&send.object, MESH_OBJ_SEND_RDMA);
    send.object.version++;
    CHECK(mesh_object_kind_of(&send) == MESH_OBJ_INVALID);
}

static void test_request_header(void) {
    struct mesh_request request;
    memset(&request, 0, sizeof(request));

    mesh_request_header_init(&request.header, MESH_OBJ_REQ_RDMA_SEND);

    CHECK(mesh_object_kind_of(&request) == MESH_OBJ_REQ_RDMA_SEND);
    CHECK(atomic_load(&request.header.state) == MESH_REQ_PENDING);
    CHECK(request.header.result == ncclSuccess);
    CHECK(request.header.completed_size == 0);

    atomic_store(&request.header.state, MESH_REQ_COMPLETE);
    request.header.completed_size = 4096;
    CHECK(mesh_request_state_is_terminal(
        atomic_load(&request.header.state)));
    CHECK(request.header.completed_size == 4096);

    atomic_store(&request.header.state, MESH_REQ_ERROR);
    request.header.result = ncclRemoteError;
    CHECK(mesh_request_state_is_terminal(
        atomic_load(&request.header.state)));
    CHECK(request.header.result == ncclRemoteError);

    atomic_store(&request.header.state, MESH_REQ_FINALIZING);
    CHECK(!mesh_request_state_is_terminal(
        atomic_load(&request.header.state)));
}

static void test_hybrid_handle_layout(void) {
    struct mesh_handle handle;
    memset(&handle, 0, sizeof(handle));

    handle.magic = MESH_HANDLE_MAGIC;
    handle.version = MESH_HANDLE_VERSION;
    handle.flags = MESH_HANDLE_FLAG_HYBRID_TCP;
    handle.hybrid_tcp_ip = 0x0100000aU;
    handle.hybrid_tcp_port = 12345;

    CHECK(sizeof(handle) <= 128);
    CHECK(handle.version == MESH_HANDLE_VERSION);
    CHECK(handle.flags & MESH_HANDLE_FLAG_HYBRID_TCP);
    CHECK(handle.hybrid_tcp_port == 12345);
}

static void test_connection_counters(void) {
    struct mesh_tcp_send_comm tcp_send;
    struct mesh_recv_comm rdma_recv;
    memset(&tcp_send, 0, sizeof(tcp_send));
    memset(&rdma_recv, 0, sizeof(rdma_recv));

    tcp_send.send_ops = 3;
    tcp_send.bytes_sent = 4096;
    rdma_recv.recv_ops = 4;
    rdma_recv.bytes_recv = 8192;

    CHECK(tcp_send.send_ops == 3);
    CHECK(tcp_send.bytes_sent == 4096);
    CHECK(rdma_recv.recv_ops == 4);
    CHECK(rdma_recv.bytes_recv == 8192);
}

static void test_tcp_request_kind(void) {
    struct mesh_tcp_request request;
    memset(&request, 0, sizeof(request));

    mesh_request_header_init(&request.header, MESH_OBJ_REQ_TCP_RECV);
    CHECK(mesh_object_kind_of(&request) == MESH_OBJ_REQ_TCP_RECV);
    CHECK(atomic_load(&request.header.state) == MESH_REQ_PENDING);
}

int main(void) {
    test_layouts();
    test_object_validation();
    test_request_header();
    test_hybrid_handle_layout();
    test_connection_counters();
    test_tcp_request_kind();

    if (failures) {
        fprintf(stderr, "%d typed-object test(s) failed\n", failures);
        return 1;
    }

    puts("typed-object tests passed");
    return 0;
}

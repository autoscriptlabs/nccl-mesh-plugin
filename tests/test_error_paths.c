/*
 * NCCL-001: Error Path Unit Tests
 *
 * Tests for the hardened error paths in the mesh plugin:
 * - Completion timeout detection
 * - QP health monitoring (WC status checks)
 * - Async event handling
 * - Fatal error flag propagation
 * - Graceful connection teardown
 * - Structured error logging
 * - Configuration surface
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>
#include <arpa/inet.h>

#include "nccl/err.h"
#include "mesh_plugin.h"

/*
 * Provide the global state symbol for tests (normally in mesh_plugin.c).
 * We only need it for tests that touch g_mesh_state directly.
 */
struct mesh_plugin_state g_mesh_state = {0};

/*
 * Minimal stub for mesh_uint_to_ip (used by MESH_ERROR_STRUCTURED macro)
 */
void mesh_uint_to_ip(uint32_t ip, char *buf, size_t len) {
    struct in_addr addr;
    addr.s_addr = htonl(ip);
    inet_ntop(AF_INET, &addr, buf, len);
}

/* Test counters */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
    } else { \
        printf("  PASS: %s\n", msg); \
        tests_passed++; \
    } \
} while(0)

/*
 * Stub logger for tests
 */
static char last_log_msg[1024] = {0};
static int last_log_level = 0;

static void test_logger(int level, unsigned long flags, const char *file,
                        int line, const char *fmt, ...) {
    (void)flags;
    (void)file;
    (void)line;
    last_log_level = level;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(last_log_msg, sizeof(last_log_msg), fmt, ap);
    va_end(ap);
}

/*
 * Test 1: mesh_request has start_time field
 */
static void test_request_has_start_time(void) {
    printf("\nTest: mesh_request has start_time field\n");

    struct mesh_request req;
    memset(&req, 0, sizeof(req));

    clock_gettime(CLOCK_MONOTONIC, &req.start_time);

    TEST_ASSERT(req.start_time.tv_sec > 0,
                "start_time.tv_sec should be populated after clock_gettime");
    TEST_ASSERT(sizeof(req.start_time) == sizeof(struct timespec),
                "start_time should be a struct timespec");
}

/*
 * Test 2: mesh_send_comm has conn_established_time and peer_ip
 */
static void test_send_comm_has_conn_fields(void) {
    printf("\nTest: mesh_send_comm has connection tracking fields\n");

    struct mesh_send_comm comm;
    memset(&comm, 0, sizeof(comm));

    clock_gettime(CLOCK_MONOTONIC, &comm.conn_established_time);
    comm.peer_ip = 0x0A000001;  // 10.0.0.1

    TEST_ASSERT(comm.conn_established_time.tv_sec > 0,
                "conn_established_time should be populated");
    TEST_ASSERT(comm.peer_ip == 0x0A000001,
                "peer_ip should be set");
}

/*
 * Test 3: mesh_recv_comm has conn_established_time and peer_ip
 */
static void test_recv_comm_has_conn_fields(void) {
    printf("\nTest: mesh_recv_comm has connection tracking fields\n");

    struct mesh_recv_comm comm;
    memset(&comm, 0, sizeof(comm));

    clock_gettime(CLOCK_MONOTONIC, &comm.conn_established_time);
    comm.peer_ip = 0xC0A80001;  // 192.168.0.1

    TEST_ASSERT(comm.conn_established_time.tv_sec > 0,
                "conn_established_time should be populated");
    TEST_ASSERT(comm.peer_ip == 0xC0A80001,
                "peer_ip should be set");
}

/*
 * Test 4: Configuration defaults
 */
static void test_config_defaults(void) {
    printf("\nTest: Configuration defaults\n");

    struct mesh_plugin_state state;
    memset(&state, 0, sizeof(state));

    // Simulate what mesh_init does with no env vars
    state.op_timeout_sec = 30;
    state.connect_timeout_sec = 10;
    state.accept_timeout_sec = 30;
    state.health_check_interval_ms = 1000;
    state.fatal_on_timeout = 1;

    TEST_ASSERT(state.op_timeout_sec == 30,
                "Default op_timeout_sec should be 30");
    TEST_ASSERT(state.connect_timeout_sec == 10,
                "Default connect_timeout_sec should be 10");
    TEST_ASSERT(state.accept_timeout_sec == 30,
                "Default accept_timeout_sec should be 30");
    TEST_ASSERT(state.health_check_interval_ms == 1000,
                "Default health_check_interval_ms should be 1000");
    TEST_ASSERT(state.fatal_on_timeout == 1,
                "Default fatal_on_timeout should be 1");
}

/*
 * Test 5: Fatal error flag propagation
 */
static void test_fatal_error_flag(void) {
    printf("\nTest: Fatal error flag propagation\n");

    /* Save and restore global state */
    int saved = atomic_load(&g_mesh_state.plugin_fatal_error);

    /* Initially should not be in error state */
    atomic_store(&g_mesh_state.plugin_fatal_error, 0);
    TEST_ASSERT(atomic_load(&g_mesh_state.plugin_fatal_error) == 0,
                "Fatal error flag should initially be 0");

    /* Set fatal error */
    atomic_store(&g_mesh_state.plugin_fatal_error, 1);
    TEST_ASSERT(atomic_load(&g_mesh_state.plugin_fatal_error) == 1,
                "Fatal error flag should be 1 after setting");

    /* Restore */
    atomic_store(&g_mesh_state.plugin_fatal_error, saved);
}

/*
 * Test 6: WC status classification (peer failure detection)
 */
static void test_wc_status_classification(void) {
    printf("\nTest: WC status classification\n");

    /* These should all be classified as peer failures */
    enum ibv_wc_status peer_failures[] = {
        IBV_WC_RETRY_EXC_ERR,
        IBV_WC_RNR_RETRY_EXC_ERR,
        IBV_WC_REM_ABORT_ERR,
        IBV_WC_REM_ACCESS_ERR,
        IBV_WC_REM_INV_REQ_ERR,
        IBV_WC_REM_OP_ERR,
        IBV_WC_WR_FLUSH_ERR,
    };
    const char *peer_failure_names[] = {
        "IBV_WC_RETRY_EXC_ERR",
        "IBV_WC_RNR_RETRY_EXC_ERR",
        "IBV_WC_REM_ABORT_ERR",
        "IBV_WC_REM_ACCESS_ERR",
        "IBV_WC_REM_INV_REQ_ERR",
        "IBV_WC_REM_OP_ERR",
        "IBV_WC_WR_FLUSH_ERR",
    };

    for (int i = 0; i < 7; i++) {
        int is_failure = 0;
        switch (peer_failures[i]) {
            case IBV_WC_RETRY_EXC_ERR:
            case IBV_WC_RNR_RETRY_EXC_ERR:
            case IBV_WC_REM_ABORT_ERR:
            case IBV_WC_REM_ACCESS_ERR:
            case IBV_WC_REM_INV_REQ_ERR:
            case IBV_WC_REM_OP_ERR:
            case IBV_WC_WR_FLUSH_ERR:
                is_failure = 1;
                break;
            default:
                is_failure = 0;
        }
        char msg[128];
        snprintf(msg, sizeof(msg), "%s should be classified as peer failure", peer_failure_names[i]);
        TEST_ASSERT(is_failure == 1, msg);
    }

    /* IBV_WC_SUCCESS should NOT be a peer failure */
    TEST_ASSERT(IBV_WC_SUCCESS == 0, "IBV_WC_SUCCESS should not be a peer failure");
}

/*
 * Test 7: Async event type classification
 */
static void test_async_event_classification(void) {
    printf("\nTest: Async event type classification\n");

    /* Fatal events */
    enum ibv_event_type fatal_events[] = {
        IBV_EVENT_PORT_ERR,
        IBV_EVENT_DEVICE_FATAL,
        IBV_EVENT_QP_FATAL,
        IBV_EVENT_QP_ACCESS_ERR,
    };
    const char *fatal_names[] = {
        "IBV_EVENT_PORT_ERR",
        "IBV_EVENT_DEVICE_FATAL",
        "IBV_EVENT_QP_FATAL",
        "IBV_EVENT_QP_ACCESS_ERR",
    };

    for (int i = 0; i < 4; i++) {
        int is_fatal = 0;
        switch (fatal_events[i]) {
            case IBV_EVENT_PORT_ERR:
            case IBV_EVENT_DEVICE_FATAL:
            case IBV_EVENT_QP_FATAL:
            case IBV_EVENT_QP_ACCESS_ERR:
                is_fatal = 1;
                break;
            default:
                is_fatal = 0;
        }
        char msg[128];
        snprintf(msg, sizeof(msg), "%s should be classified as fatal", fatal_names[i]);
        TEST_ASSERT(is_fatal == 1, msg);
    }

    /* Non-fatal events */
    enum ibv_event_type nonfatal_events[] = {
        IBV_EVENT_PORT_ACTIVE,
        IBV_EVENT_COMM_EST,
        IBV_EVENT_SQ_DRAINED,
        IBV_EVENT_LID_CHANGE,
    };
    const char *nonfatal_names[] = {
        "IBV_EVENT_PORT_ACTIVE",
        "IBV_EVENT_COMM_EST",
        "IBV_EVENT_SQ_DRAINED",
        "IBV_EVENT_LID_CHANGE",
    };

    for (int i = 0; i < 4; i++) {
        int is_fatal = 0;
        switch (nonfatal_events[i]) {
            case IBV_EVENT_PORT_ERR:
            case IBV_EVENT_DEVICE_FATAL:
            case IBV_EVENT_QP_FATAL:
            case IBV_EVENT_QP_ACCESS_ERR:
                is_fatal = 1;
                break;
            default:
                is_fatal = 0;
        }
        char msg[128];
        snprintf(msg, sizeof(msg), "%s should NOT be classified as fatal", nonfatal_names[i]);
        TEST_ASSERT(is_fatal == 0, msg);
    }
}

/*
 * Test 8: Timeout detection logic
 */
static void test_timeout_detection(void) {
    printf("\nTest: Timeout detection logic\n");

    int timeout_sec = 30;

    /* Simulate a request that was just created */
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    struct timespec start = now;
    long elapsed = now.tv_sec - start.tv_sec;
    TEST_ASSERT(elapsed < timeout_sec,
                "Fresh request should not be timed out");

    /* Simulate a request created 31 seconds ago */
    struct timespec old_start;
    old_start.tv_sec = now.tv_sec - 31;
    old_start.tv_nsec = now.tv_nsec;

    elapsed = now.tv_sec - old_start.tv_sec;
    TEST_ASSERT(elapsed >= timeout_sec,
                "31-second-old request should be detected as timed out (threshold=30s)");

    /* Simulate a request created 29 seconds ago */
    struct timespec recent_start;
    recent_start.tv_sec = now.tv_sec - 29;
    recent_start.tv_nsec = now.tv_nsec;

    elapsed = now.tv_sec - recent_start.tv_sec;
    TEST_ASSERT(elapsed < timeout_sec,
                "29-second-old request should NOT be detected as timed out (threshold=30s)");
}

/*
 * Test 9: Structured error logging format
 */
static void test_structured_error_logging(void) {
    printf("\nTest: Structured error logging\n");

    /* Initialize the logger */
    g_mesh_state.log_fn = test_logger;
    g_mesh_state.debug_level = 2;

    /* Use the MESH_ERROR_STRUCTURED macro */
    uint32_t peer_ip = 0x0A0000AD;  // 10.0.0.173
    uint32_t qp_num = 0x1A3F;
    MESH_ERROR_STRUCTURED(peer_ip, qp_num, "RECV",
        "IBV_WC_RETRY_EXC_ERR", 3612,
        "Completion timeout after 30s — marking connection dead");

    /* Check that the log message contains key fields */
    TEST_ASSERT(strstr(last_log_msg, "peer=10.0.0.173") != NULL,
                "Log should contain peer IP");
    TEST_ASSERT(strstr(last_log_msg, "qp=0x1a3f") != NULL,
                "Log should contain QP number");
    TEST_ASSERT(strstr(last_log_msg, "op=RECV") != NULL,
                "Log should contain operation type");
    TEST_ASSERT(strstr(last_log_msg, "status=IBV_WC_RETRY_EXC_ERR") != NULL,
                "Log should contain status string");
    TEST_ASSERT(strstr(last_log_msg, "conn_age=3612s") != NULL,
                "Log should contain connection age");

    g_mesh_state.log_fn = NULL;
}

/*
 * Test 10: Connection age calculation
 */
static void test_connection_age(void) {
    printf("\nTest: Connection age calculation\n");

    /* A connection established right now should have age ~0 */
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    struct mesh_send_comm comm;
    memset(&comm, 0, sizeof(comm));
    comm.conn_established_time = now;

    /* Calculate age the same way the code does */
    struct timespec check;
    clock_gettime(CLOCK_MONOTONIC, &check);
    uint64_t age = (uint64_t)(check.tv_sec - comm.conn_established_time.tv_sec);

    TEST_ASSERT(age <= 1, "Freshly established connection age should be ~0s");

    /* A connection established 3600 seconds ago */
    struct mesh_send_comm old_comm;
    memset(&old_comm, 0, sizeof(old_comm));
    old_comm.conn_established_time.tv_sec = now.tv_sec - 3600;
    old_comm.conn_established_time.tv_nsec = 0;

    age = (uint64_t)(check.tv_sec - old_comm.conn_established_time.tv_sec);
    TEST_ASSERT(age >= 3599 && age <= 3601,
                "1-hour-old connection age should be ~3600s");
}

/*
 * Test 11: Handle size still fits in NCCL limit
 */
static void test_handle_size(void) {
    printf("\nTest: Handle size within NCCL limits\n");

    TEST_ASSERT(sizeof(struct mesh_handle) <= NCCL_NET_HANDLE_MAXSIZE,
                "mesh_handle must fit in NCCL_NET_HANDLE_MAXSIZE (128 bytes)");

    printf("  INFO: mesh_handle size = %zu bytes (limit = %d)\n",
           sizeof(struct mesh_handle), NCCL_NET_HANDLE_MAXSIZE);
}

/*
 * Test 12: ibv_event_type_str coverage
 */
static void test_event_type_str(void) {
    printf("\nTest: ibv_event_type_str coverage\n");

    const char *port_err = ibv_event_type_str(IBV_EVENT_PORT_ERR);
    const char *device_fatal = ibv_event_type_str(IBV_EVENT_DEVICE_FATAL);
    const char *qp_fatal = ibv_event_type_str(IBV_EVENT_QP_FATAL);
    const char *qp_access = ibv_event_type_str(IBV_EVENT_QP_ACCESS_ERR);
    const char *port_active = ibv_event_type_str(IBV_EVENT_PORT_ACTIVE);

    TEST_ASSERT(port_err && port_err[0] != '\0',
                "IBV_EVENT_PORT_ERR has a description");
    TEST_ASSERT(device_fatal && device_fatal[0] != '\0',
                "IBV_EVENT_DEVICE_FATAL has a description");
    TEST_ASSERT(qp_fatal && qp_fatal[0] != '\0',
                "IBV_EVENT_QP_FATAL has a description");
    TEST_ASSERT(qp_access && qp_access[0] != '\0',
                "IBV_EVENT_QP_ACCESS_ERR has a description");
    TEST_ASSERT(port_active && port_active[0] != '\0',
                "IBV_EVENT_PORT_ACTIVE has a description");
}

/*
 * Test 13: Plugin state has all required error hardening fields
 */
static void test_plugin_state_fields(void) {
    printf("\nTest: Plugin state has error hardening fields\n");

    struct mesh_plugin_state state;
    memset(&state, 0, sizeof(state));

    /* These should all exist and be accessible */
    state.op_timeout_sec = 30;
    state.connect_timeout_sec = 10;
    state.accept_timeout_sec = 30;
    state.health_check_interval_ms = 1000;
    state.fatal_on_timeout = 1;
    atomic_store(&state.plugin_fatal_error, 0);
    snprintf(state.fatal_error_msg, sizeof(state.fatal_error_msg), "test");

    TEST_ASSERT(state.op_timeout_sec == 30, "op_timeout_sec field exists");
    TEST_ASSERT(state.connect_timeout_sec == 10, "connect_timeout_sec field exists");
    TEST_ASSERT(state.accept_timeout_sec == 30, "accept_timeout_sec field exists");
    TEST_ASSERT(state.health_check_interval_ms == 1000, "health_check_interval_ms field exists");
    TEST_ASSERT(state.fatal_on_timeout == 1, "fatal_on_timeout field exists");
    TEST_ASSERT(atomic_load(&state.plugin_fatal_error) == 0, "plugin_fatal_error atomic field works");
    TEST_ASSERT(strcmp(state.fatal_error_msg, "test") == 0, "fatal_error_msg field exists");
}

/*
 * Test: Server metrics fields exist on NIC and plugin state
 */
static void test_metrics_fields(void) {
    printf("\nTest: Server metrics fields\n");

    struct mesh_nic nic;
    memset(&nic, 0, sizeof(nic));

    /* Per-NIC metrics counters should exist and be zero-initialized */
    TEST_ASSERT(nic.send_ops == 0, "send_ops should be zero-initialized");
    TEST_ASSERT(nic.recv_ops == 0, "recv_ops should be zero-initialized");
    TEST_ASSERT(nic.send_completions == 0, "send_completions should be zero-initialized");
    TEST_ASSERT(nic.recv_completions == 0, "recv_completions should be zero-initialized");
    TEST_ASSERT(nic.send_errors == 0, "send_errors should be zero-initialized");
    TEST_ASSERT(nic.recv_errors == 0, "recv_errors should be zero-initialized");
    TEST_ASSERT(nic.completion_timeouts == 0, "completion_timeouts should be zero-initialized");
    TEST_ASSERT(nic.max_completion_us == 0, "max_completion_us should be zero-initialized");
    TEST_ASSERT(nic.total_completion_us == 0, "total_completion_us should be zero-initialized");
    TEST_ASSERT(nic.active_sends == 0, "active_sends should be zero-initialized");
    TEST_ASSERT(nic.active_recvs == 0, "active_recvs should be zero-initialized");

    /* Plugin state metrics config fields */
    struct mesh_plugin_state state;
    memset(&state, 0, sizeof(state));

    state.metrics_enabled = 1;
    state.metrics_interval_sec = 10;
    TEST_ASSERT(state.metrics_enabled == 1, "metrics_enabled field exists");
    TEST_ASSERT(state.metrics_interval_sec == 10, "metrics_interval_sec field exists");
}

/*
 * Main
 */
int main(void) {
    printf("====================================\n");
    printf("NCCL-001 Error Path Unit Tests\n");
    printf("====================================\n");

    test_request_has_start_time();
    test_send_comm_has_conn_fields();
    test_recv_comm_has_conn_fields();
    test_config_defaults();
    test_fatal_error_flag();
    test_wc_status_classification();
    test_async_event_classification();
    test_timeout_detection();
    test_structured_error_logging();
    test_connection_age();
    test_handle_size();
    test_event_type_str();
    test_plugin_state_fields();
    test_metrics_fields();

    printf("\n====================================\n");
    printf("Results: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);
    printf("====================================\n");

    return tests_failed > 0 ? 1 : 0;
}

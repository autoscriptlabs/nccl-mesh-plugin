# NCCL Mesh Plugin Makefile

CC = gcc
PKG_CONFIG ?= pkg-config

# Keep project quote-includes separate so angle-bracket system headers,
# including <infiniband/verbs.h>, resolve from the system include paths.
IBVERBS_CFLAGS := $(shell $(PKG_CONFIG) --cflags libibverbs 2>/dev/null)
IBVERBS_LIBS := $(shell $(PKG_CONFIG) --libs libibverbs 2>/dev/null)
ifeq ($(strip $(IBVERBS_LIBS)),)
IBVERBS_LIBS := -libverbs
endif

CFLAGS = -Wall -Wextra -O2 -fPIC -g
CFLAGS += -I. -iquote ./include $(IBVERBS_CFLAGS)
LDFLAGS = -shared
LDLIBS = $(IBVERBS_LIBS) -lpthread -ldl

# Target
TARGET = libnccl-net.so
TARGET_MESH = libnccl-net-mesh.so

# Sources
SRCS = src/mesh_plugin.c src/mesh_routing.c
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET) $(TARGET_MESH)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

$(TARGET_MESH): $(TARGET)
	ln -sf $(TARGET) $(TARGET_MESH)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Install to a standard location
PREFIX ?= /usr/local
install: all
	install -d $(PREFIX)/lib
	install -m 755 $(TARGET) $(PREFIX)/lib/
	ln -sf $(TARGET) $(PREFIX)/lib/$(TARGET_MESH)

# Clean
clean:
	rm -f $(OBJS) $(TARGET) $(TARGET_MESH) tests/test_routing tests/test_error_paths

# Test build (requires libibverbs-dev)
test-deps:
	@echo "Checking dependencies..."
	@pkg-config --exists libibverbs || (echo "ERROR: libibverbs-dev not found" && exit 1)
	@echo "All dependencies found."

# Debug build
debug: CFLAGS += -DDEBUG -g3 -O0
debug: clean all

# Print configuration
info:
	@echo "CC      = $(CC)"
	@echo "CFLAGS  = $(CFLAGS)"
	@echo "LDFLAGS = $(LDFLAGS)"
	@echo "LDLIBS  = $(LDLIBS)"
	@echo "TARGET  = $(TARGET)"

# ============================================================================
# Test Targets
# ============================================================================

# Test flags (no -fPIC, add pthread)
TEST_CFLAGS = -Wall -Wextra -O2 -g -I. -iquote ./include $(IBVERBS_CFLAGS)
TEST_LDFLAGS = -lpthread

# Unit test for routing
test_routing: tests/test_routing.c src/mesh_routing.c
	$(CC) $(TEST_CFLAGS) $^ -o tests/$@ $(TEST_LDFLAGS)

# Unit test for error paths (NCCL-001)
test_error_paths: tests/test_error_paths.c
	$(CC) $(TEST_CFLAGS) $^ -o tests/$@ $(TEST_LDFLAGS)

# Run unit tests
test: test_routing test_error_paths
	@echo ""
	@echo "Running unit tests..."
	@./tests/test_routing
	@./tests/test_error_paths
	@echo ""
	@echo "Running integration tests..."
	@python3 tests/test_ring_topo.py
	@python3 tests/test_line_topo.py

# Run unit tests only (C)
test-unit: test_routing test_error_paths
	@./tests/test_routing
	@./tests/test_error_paths

# Run integration tests only (Python)
test-integration:
	@python3 tests/test_ring_topo.py -v
	@python3 tests/test_line_topo.py -v

.PHONY: all clean install test-deps debug info test test_routing test_error_paths test-unit test-integration

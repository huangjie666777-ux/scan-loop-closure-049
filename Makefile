CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude -Ithird_party/eigen3

BUILD := build
LIB := $(BUILD)/libicp3d.a
SRCS := src/kdtree.cpp src/rigid.cpp src/types.cpp src/icp.cpp src/se3.cpp src/pose_graph_optimizer.cpp src/point_cloud_map.cpp src/multiscan.cpp
OBJS := $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRCS))

.PHONY: all test example clean

all: $(LIB) $(BUILD)/icp_example $(BUILD)/multiscan_example $(BUILD)/icp_tests

$(LIB): $(OBJS)
	@mkdir -p $(BUILD)
	ar rcs $@ $^

$(BUILD)/%.o: src/%.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Eigen 3.4 的 JacobiSVD 在 GCC 11 -O2 下有已知 maybe-uninitialized 误报。
$(BUILD)/rigid.o: CXXFLAGS += -Wno-maybe-uninitialized

$(BUILD)/icp_example: examples/example.cpp $(LIB)
	$(CXX) $(CXXFLAGS) $< -L$(BUILD) -licp3d -o $@

$(BUILD)/multiscan_example: examples/multiscan_example.cpp $(LIB)
	$(CXX) $(CXXFLAGS) $< -L$(BUILD) -licp3d -o $@

$(BUILD)/icp_tests: tests/test_icp.cpp $(LIB)
	$(CXX) $(CXXFLAGS) $< -L$(BUILD) -licp3d -o $@

test: $(BUILD)/icp_tests
	./$(BUILD)/icp_tests

example: $(BUILD)/icp_example $(BUILD)/multiscan_example
	./$(BUILD)/icp_example
	./$(BUILD)/multiscan_example

clean:
	rm -rf $(BUILD)

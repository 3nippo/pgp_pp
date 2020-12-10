CC = nvcc

FLAGS = -gencode arch=compute_50,code=sm_50 -ccbin=mpic++ --std=c++14 -Werror cross-execution-space-call -lm
DEBUG =

SOURCES = main.cu lab08.cu
BIN = lab08

TEST_SUFFIX = test
TEST_SOURCES = test.cu lab08.cu lab08cpu.cpp
TEST_FLAGS = -lgtest -lgmock -lgtest_main

all:
	$(CC) $(DEBUG) $(FLAGS) -o $(BIN) $(SOURCES)
test:
	$(CC) $(DEBUG) $(FLAGS) -o $(addsuffix $(TEST_SUFFIX), $(BIN)) $(TEST_SOURCES) $(TEST_FLAGS)

clean:
	rm -rf $(BIN) $(addsuffix $(TEST_SUFFIX), $(BIN)) mpi.out cpu.out

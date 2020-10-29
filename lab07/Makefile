CC = mpic++

FLAGS = -std=c++11 -O3
DEBUG =

SOURCES = main.cpp lab07.cpp
BIN = lab07

TEST_SUFFIX = test
TEST_SOURCES = test.cpp lab07.cpp lab07cpu.cpp
TEST_FLAGS = -lgtest -lgmock -lgtest_main

all:
	$(CC) $(DEBUG) $(FLAGS) -o $(BIN) $(SOURCES)
test:
	$(CC) $(DEBUG) $(FLAGS) -o $(addsuffix $(TEST_SUFFIX), $(BIN)) $(TEST_SOURCES) $(TEST_FLAGS)

clean:
	rm -rf $(BIN) $(addsuffix $(TEST_SUFFIX), $(BIN)) mpi.out cpu.out

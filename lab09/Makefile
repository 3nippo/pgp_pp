CC = mpic++

FLAGS = -std=c++11 -fopenmp -O3 -pedantic -Wall -Werror -Wno-sign-compare -Wno-long-long -lm
DEBUG =

SOURCES = main.cpp lab09.cpp
BIN = lab09

TEST_SUFFIX = test
TEST_SOURCES = test.cpp lab09.cpp lab09cpu.cpp
TEST_FLAGS = -lgtest -lgmock -lgtest_main

all:
	$(CC) $(DEBUG) $(FLAGS) -o $(BIN) $(SOURCES)
test:
	$(CC) $(DEBUG) $(FLAGS) -o $(addsuffix $(TEST_SUFFIX), $(BIN)) $(TEST_SOURCES) $(TEST_FLAGS)

clean:
	rm -rf $(BIN) $(addsuffix $(TEST_SUFFIX), $(BIN)) mpi.out cpu.out

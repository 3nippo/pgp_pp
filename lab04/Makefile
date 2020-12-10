CC = nvcc
FLAGS = -O3 -gencode arch=compute_50,code=sm_50 -std=c++14 -Werror cross-execution-space-call -lm

OBJECTS = lab04.o main.o

BIN = lab04

%.o: %.cu
	$(CC) -x cu $(FLAGS) -I. -dc $< -o $@

all: $(OBJECTS)
	$(CC) $(FLAGS) $(OBJECTS) -o $(BIN)

clean:
	rm -rf $(BIN) *.o

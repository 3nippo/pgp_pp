CC = nvcc
FLAGS = -gencode arch=compute_50,code=sm_50 -std=c++11 -Werror cross-execution-space-call -lm

OBJECTS = lab05.o main.o

BIN = lab05

%.o: %.cu
	$(CC) -x cu $(FLAGS) -I. -dc $< -o $@

all: $(OBJECTS)
	$(CC) $(FLAGS) $(OBJECTS) -o $(BIN)

clean:
	rm -rf $(BIN) *.o

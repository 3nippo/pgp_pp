CC = nvcc
FLAGS = -gencode arch=compute_50,code=sm_50

OBJECTS = lab03.o math.o

BIN = lab03

%.o: %.cu
	$(CC) -x cu $(FLAGS) -I. -dc $< -o $@

all: $(OBJECTS)
	$(CC) $(FLAGS) $(OBJECTS) -o $(BIN)

clean:
	rm -rf $(BIN) *.data input *.o

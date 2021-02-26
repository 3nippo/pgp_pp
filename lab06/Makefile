CC = nvcc
FLAGS = -gencode arch=compute_50,code=sm_50 -std=c++11 -lGL -lGLU -lglut -lGLEW

OBJECTS = main.o lab06.o

BIN = lab06

%.o: %.cu
	$(CC) -x cu $(FLAGS) -I. -dc $< -o $@

all: $(OBJECTS)
	$(CC) $(FLAGS) $(OBJECTS) -o $(BIN)

clean:
	rm -rf $(BIN) *.o

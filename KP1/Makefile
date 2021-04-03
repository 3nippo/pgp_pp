CC = g++
# FLAGS = -O3 -gencode arch=compute_50,code=sm_50 -std=c++14 -Werror cross-execution-space-call -lm
FLAGS = -std=c++11
DEBUG = -g

OBJECTS = Camera.o FigureFacesConstructor.o main.o RayTracer.o SquareFace.o TriangleFace.o utils.o Vector3.o

BIN = RayTracer

%.o: %.cpp
	$(CC) $(DEBUG) $(FLAGS) -I. -dc $< -o $@

all: $(OBJECTS)
	$(CC) $(DEBUG) $(FLAGS) $(OBJECTS) -o $(BIN)

clean:
	rm -rf $(BIN) *.o

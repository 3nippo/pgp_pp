CC = nvcc
FLAGS = -gencode arch=compute_50,code=sm_50 -std=c++14

OBJECTS = Camera.o FigureFacesConstructor.o main.o RayTracer.o SquareFace.o TriangleFace.o utils.o Vector3.o Image.o

BIN = RayTracer

%.o: %.cu
	$(CC) -x cu $(FLAGS) -I. -dc $< -o $@

all: $(OBJECTS)
	$(CC) $(FLAGS) $(OBJECTS) -o $(BIN)

clean:
	rm -rf $(BIN) *.o

CC = g++
# FLAGS = -O3 -gencode arch=compute_50,code=sm_50 -std=c++14 -Werror cross-execution-space-call -lm
FLAGS = -std=c++11
DEBUG = -g

SOURCES = Camera.cpp FigureFacesConstructor.cpp main.cpp RayTracer.cpp SquareFace.cpp TriangleFace.cpp utils.cpp Vector3.cpp

BIN = RayTracer

all:
	$(CC) $(DEBUG) $(FLAGS) -o $(BIN) $(SOURCES)

clean:
	rm -rf $(BIN) *.o

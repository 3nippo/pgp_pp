CC = nvcc
FLAGS = -g -G -gencode arch=compute_50,code=sm_50
SOURCES = lab03.cu
BIN = lab03
all:
	$(CC) $(FLAGS) -o $(BIN) $(SOURCES)

clean:
	rm -rf $(BIN) *.data input

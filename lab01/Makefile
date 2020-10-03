CC = nvcc
FLAGS = -gencode arch=compute_50,code=sm_50
SOURCES = lab01.cu
BIN = lab01
all:
	$(CC) $(FLAGS) -o $(BIN) $(SOURCES)

clean:
	rm -rf $(BIN) output

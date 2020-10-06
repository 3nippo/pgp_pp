CC = nvcc
FLAGS = -gencode arch=compute_50,code=sm_50
SOURCES = lab02.cu
BIN = lab02
all:
	$(CC) $(FLAGS) -o $(BIN) $(SOURCES)

clean:
	rm -rf $(BIN) *.data

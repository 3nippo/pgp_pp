import numpy as np
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


lab_num = 8
input_name = 'input'


def benchmark():
    def set_grid_and_block_sizes(grid_size, block_size):
        source_name = "lab0{}.cu".format(lab_num)

        with open(source_name) as f:
            lines = f.readlines()

        with open(source_name, 'w') as f:
            for line in lines:
                if line.startswith("#define GRID_SIZE "):
                    f.write("#define GRID_SIZE {}\n".format(grid_size))
                elif line.startswith("#define BLOCK_SIZE "):
                    f.write("#define BLOCK_SIZE {}\n".format(block_size))
                else:
                    f.write(line)

    grid_sizes = [i for i in range(1, 16+1)]
    block_sizes = [4, 8]

    rows = []

    for grid_size in grid_sizes:
        row = []

        for block_size in block_sizes:
            print(grid_size, block_size)

            set_grid_and_block_sizes(grid_size, block_size)

            compilation_process = subprocess.Popen(
                "make".split()
            )

            compilation_process.wait()

            with open(input_name, 'rb') as f:
                content = f.read()

            gpu_process = subprocess.Popen(
                "mpirun -hostfile hostfile -np 8 ./lab0{}".format(lab_num).split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                shell=True
            )

            output_gpu, _ = gpu_process.communicate(input=content)
            row.append(float(output_gpu.decode().strip().split('\n')[-1].split()[1]))

        rows.append(row)

    df = pd.DataFrame(
        rows,
        index=[str(el**3) for el in grid_sizes],
        columns=[str(el**3) for el in block_sizes]
    )

    plt.figure(figsize=(10, 10))

    sns.heatmap(df, cmap="YlGnBu", fmt='4f', annot=True, cbar_kws={'label': 'Время, мс'})

    plt.xlabel("Количество потоков в блоке")
    plt.ylabel("Количество блоков")

    plt.savefig('lab0{}.pdf'.format(lab_num))


benchmark()

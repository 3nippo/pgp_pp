import numpy as np
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


lab_num = 1


def gen_data():
    n = 2**25 - 1
    # n = 50
    N = n

    a = np.random.randint(-n, n, N) * np.random.uniform(0, 1, N)
    b = np.random.randint(-n, n, N) * np.random.uniform(0, 1, N)

    a_str = " ".join(map(lambda x: "%.10e" % x, a))
    b_str = " ".join(map(lambda x: "%.10e" % x, b))

    text_input = "\n".join([str(N), a_str, b_str])

    with open("input", 'w') as f:
        f.write(text_input)


gen_data()


def run_cpu():
    with open("input") as f:
        content = f.read()

    cpu_process = subprocess.Popen(
        "./lab0{}_cpu".format(lab_num).split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    output_cpu, _ = cpu_process.communicate(input=content.encode())

    print("CPU {}ms".format(output_cpu.decode().split()[1]))


def benchmark():
    def set_grid_and_block_sizes(grid_size, block_size):
        source_name = "lab0{}.cu".format(lab_num)

        with open(source_name) as f:
            lines = f.readlines()

        with open(source_name, 'w') as f:
            for line in lines:
                if "#define GRID_SIZE" in line:
                    f.write("#define GRID_SIZE {}\n".format(grid_size))
                elif "#define BLOCK_SIZE" in line:
                    f.write("#define BLOCK_SIZE {}\n".format(block_size))
                else:
                    f.write(line)

    grid_size = 1

    rows = []

    while grid_size <= 1024:
        block_size = 32
        row = []

        while block_size <= 1024:
            set_grid_and_block_sizes(grid_size, block_size)

            compilation_process = subprocess.Popen(
                "make".split()
            )

            compilation_process.wait()

            with open('input') as f:
                content = f.read()

            gpu_process = subprocess.Popen(
                "./lab0{} < input | grep time:".format(lab_num).split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE
            )

            output_gpu, _ = gpu_process.communicate(input=content.encode())

            row.append(float(output_gpu.decode().split()[1]))

            block_size *= 2

        rows.append(row)

        grid_size *= 2

    df = pd.DataFrame(
        rows,
        index=[str(2**i) for i in range(11)],
        columns=[str(2**i) for i in range(5, 11)]
    )

    sns.heatmap(df, cmap="YlGnBu", fmt='4f', annot=True, cbar_kws={'label': 'Время, мс'})

    plt.xlabel("Количество потоков в блоке")
    plt.ylabel("Количество блоков")

    plt.show()


run_cpu()
benchmark()

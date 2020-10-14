import numpy as np
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


lab_num = 2


def gen_data():
    # w = 10
    # h = 10

    w = 10**4
    h = 10**4

    image = np.random.randint(0, 2**32, w * h)

    with open("in.data", 'wb') as f:
        f.write(w.to_bytes(4, 'little'))
        f.write(h.to_bytes(4, 'little'))

        for pixel in image:
            f.write(int(pixel).to_bytes(4, 'little'))

    with open('input', 'w') as f:
        f.write('in.data\n')
        f.write('out.data\n')


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

    MAX_GRID_SIZE = 32
    MAX_BLOCK_SIZE = 32
    BLOCK_START = 6
    grid_size = 1

    rows = []

    while grid_size <= MAX_GRID_SIZE:
        block_size = BLOCK_START
        row = []

        while block_size <= MAX_BLOCK_SIZE:
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

            if block_size == BLOCK_START:
                block_size = 8
            else:
                block_size *= 2

        rows.append(row)

        grid_size *= 2

    df = pd.DataFrame(
        rows,
        index=[str(2**i) for i in range(6)],
        columns=[36, 64, 256, 1024]
    )

    plt.figure(figsize=(10, 10))

    sns.heatmap(df, cmap="YlGnBu", fmt='4f', annot=True, cbar_kws={'label': 'Время, мс'})

    plt.xlabel("Количество потоков в блоке")
    plt.ylabel("Количество блоков")

    plt.savefig('lab0{}.pdf'.format(lab_num))


run_cpu()
benchmark()

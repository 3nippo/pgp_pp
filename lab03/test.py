import numpy as np
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


lab_num = 3
input_name = 'in.data'
output_name = 'out.data'
stdin_name = 'input'


def gen_data():
    w = 2*10**4
    h = 2*10**4

    # image = [2**32 - 1 for _ in range(w * h)]
    image = np.random.randint(0, 2**32, w * h)

    with open(input_name, 'wb') as f:
        f.write(w.to_bytes(4, 'little'))
        f.write(h.to_bytes(4, 'little'))

        for pixel in image:
            f.write(int(pixel).to_bytes(4, 'little'))

    with open(stdin_name, 'w') as f:
        nc = np.random.randint(0, 33)

        print(input_name, file=f)
        print(output_name, file=f)

        print(nc, file=f)

        for _ in range(nc):
            # npj = w*h
            npj = 2**19

            choosed_pixel_positions = []

            for _ in range(npj):
                pixel_position = (np.random.randint(0, h), np.random.randint(0, w))

                choosed_pixel_positions.append(pixel_position)

            single_coordinates = []

            for pixel_position in choosed_pixel_positions:
                single_coordinates.extend(pixel_position)

            sample_line = " ".join([str(npj)] + list(map(str, single_coordinates)))

            print(sample_line, file=f)


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

    MAX_GRID_SIZE = 1024
    MAX_BLOCK_SIZE = 1024
    BLOCK_START = 32
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

            block_size *= 2

        rows.append(row)

        grid_size *= 2

    df = pd.DataFrame(
        rows,
        index=[str(2**i) for i in range(11)],
        columns=[str(2**i) for i in range(5, 11)]
    )

    plt.figure(figsize=(10, 10))

    sns.heatmap(df, cmap="YlGnBu", fmt='4f', annot=True, cbar_kws={'label': 'Время, мс'})

    plt.xlabel("Количество потоков в блоке")
    plt.ylabel("Количество блоков")

    plt.savefig('lab0{}.pdf'.format(lab_num))


run_cpu()
benchmark()

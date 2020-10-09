import sys
import math
import numpy as np
import subprocess

tests_num = 1
input_name = 'in.data'
stdin_name = 'input'
output_name = 'out.data'
python_output_name = 'py_out.data'

cmd = './lab02 < input'


def report_error(test_index, error_code, error):
    test_index += 1

    print("Test #{}".format(test_index))
    print("Input is in{}.data".format(test_index))
    print("Error code: {}".format(error_code))
    print("Error message:")
    print(error.decode().strip())
    print()

    cmd = "cp in.data in{}.data".format(test_index)
    process = subprocess.Popen(cmd.split())
    process.wait()

    cmd = "cp input input{}".format(test_index)
    process = subprocess.Popen(cmd.split())
    process.wait()


def generate_data():
    w = np.random.randint(0, 4 * 10**8 // 10000)
    h = 4*10**8 // w

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

        used_positions = set()

        for _ in range(nc):
            npj = np.random.randint(0, 2**19 + 1)

            choosed_pixel_positions = []

            for _ in range(npj):
                while (pixel_position := (np.random.randint(0, h), np.random.randint(0, w))) in used_positions:
                    pass

                choosed_pixel_positions.append(pixel_position)

                used_positions.add(pixel_position)

            single_coordinates = []

            for pixel_position in choosed_pixel_positions:
                single_coordinates.extend(pixel_position)

            sample_line = " ".join([str(npj)] + list(map(str, single_coordinates)))

            print(sample_line, file=f)


def read_data():
    with open(input_name, 'rb') as f:
        w = f.read(4)
        h = f.read(4)

        w, h = map(lambda x: int.from_bytes(x, 'little'), [w, h])

        image = []

        for i in range(h):
            image_row = []

            for j in range(w):
                r = f.read(1)
                g = f.read(1)
                b = f.read(1)
                _ = f.read(1)

                r, g, b = map(lambda x: int.from_bytes(x, 'little'), [r, g, b])

                image_row.append([r, g, b])

            image.append(image_row)

    with open(stdin_name) as f:
        _ = f.readline()
        _ = f.readline()

        nc = int(f.readline().strip())

        npjs = []

        for _ in range(nc):
            sample_line = map(int, f.readline().strip().split())

            npjs.append(list(sample_line))

    return image, nc, npjs, h, w


def process_data(image, nc, npjs, h, w):
    avg = []
    cov_matrices = []

    for i in range(nc):
        sum_vec = np.array((0, 0, 0), dtype=np.float32).reshape((-1, 1))

        npj = npjs[i]

        for j in range(1, len(npj), 2):
            sum_vec += np.array(image[npj[j+1]][npj[j]], dtype=np.float32).reshape((-1, 1))

        sum_vec = sum_vec / npj[0]

        avg.append(sum_vec)

        cov_matrix = np.zeros((3, 3), dtype=np.float32)

        for j in range(1, len(npj), 2):
            pixel = np.array(image[npj[j+1]][npj[j]], dtype=np.float32).reshape((-1, 1))
            cov_matrix += (pixel - avg[-1]) @ (pixel - avg[-1]).T

        cov_matrix = cov_matrix / (npj[0] - 1)

        cov_matrices.append(cov_matrix)

    for i in range(h):
        for j in range(w):
            pixel = np.array(image[i][j]).reshape((-1, 1))

            mmp = [
                -((pixel - avg[i]).T @ np.linalg.inv(cov_matrices[i]) @ (pixel - avg[i]) + np.log(np.linalg.norm(cov_matrices[i], float('inf'))))[0][0]
                for i in range(nc)
            ]

            pixel_class = np.argmax(mmp)

            image[i][j].append(pixel_class)

    return image, h, w


def write_data(result, h, w):
    with open(python_output_name, 'wb') as f:
        f.write(int(w).to_bytes(4, 'little'))
        f.write(int(h).to_bytes(4, 'little'))
        # f.write(int(0).to_bytes(4, 'little'))

        for i in range(h):
            for j in range(w):
                r, g, b, a = result[i][j]

                f.write(int(r).to_bytes(1, 'little'))
                f.write(int(g).to_bytes(1, 'little'))
                f.write(int(b).to_bytes(1, 'little'))
                f.write(int(a).to_bytes(1, 'little'))


gen_data = 'generate' in sys.argv
dont_gen_data = 'dont_generate' in sys.argv

for t in range(tests_num):
    if not dont_gen_data:
        generate_data()

    if gen_data:
        break

    args = read_data()

    result = process_data(*args)

    write_data(*result)
    break
    # run C++ solution
    process = subprocess.Popen(
        cmd.split(),
        stderr=subprocess.PIPE
    )

    _, error = process.communicate()

    if process.returncode != 0 or error:
        report_error(t, process.returncode, error)

    # cmp C++ solution with Python solution
    cmp_cmd = "cmp {} {}".format(output_name, python_output_name)

    process = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    output, error = process.communicate()

    if process.returncode != 0 or error or output:
        report_error(t, process.returncode, error)

import math
import numpy as np
import subprocess

tests_num = 1
input_name = 'in.data'
output_name = 'out.data'
python_output_name = 'py_out.data'

cmd = './lab02'


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


def generate_data():
    w = np.random.randint(0, 10**8 + 1)
    h = 10**8 // w

    image = np.random.randint(0, 2**32, w * h)

    with open(input_name, 'wb') as f:
        f.write(w.to_bytes(4, 'little'))
        f.write(h.to_bytes(4, 'little'))

        for pixel in image:
            f.write(int(pixel).to_bytes(4, 'little'))


def read_data():
    with open(input_name, 'rb') as f:
        w = f.read(4)
        h = f.read(4)

        w, h = map(lambda x: int.from_bytes(x, 'little'), [w, h])

        m = []

        for i in range(w*h):
            r = f.read(1)
            g = f.read(1)
            b = f.read(1)
            _ = f.read(1)

            r, g, b = map(lambda x: int.from_bytes(x, 'little'), [r, g, b])

            y = 0.299 * r + 0.587 * g + 0.114 * b
            # u = -0.168736 * r + -0.331264 * g + 0.5 * b + 128
            # v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128

            m.append(y)

    m = [m[i:i+w] for i in range(0, w*h, w)]
    m = [row + [row[-1]] for row in m]
    m = m + [m[-1]]

    return m, (w, h)


def process_data(m, w, h):
    new_m = []

    for i in range(h):
        for j in range(w):
            Gx = m[i][j] - m[i+1][j+1]
            Gy = m[i][j+1] - m[i+1][j]
            G = math.sqrt(Gx*Gx + Gy*Gy)
            # G = abs(Gx) + abs(Gy)
            new_m.append(G)

    return [new_m[i:i+w] for i in range(0, w*h, w)]


def write_data(m, w, h):
    with open(python_output_name, 'wb') as f:
        f.write(int(w).to_bytes(4, 'little'))
        f.write(int(h).to_bytes(4, 'little'))
        # f.write(int(0).to_bytes(4, 'little'))

        for i in range(h):
            for j in range(w):
                r = min(m[i][j], 255)
                g = min(m[i][j], 255)
                b = min(m[i][j], 255)

                # r = y + 1.402 * (v - 128)
                # g = y - 0.344136 * (u - 128) - 0.714136 * (v - 128)
                # b = y + 1.772 * (u - 128)

                f.write(int(r).to_bytes(1, 'little'))
                f.write(int(g).to_bytes(1, 'little'))
                f.write(int(b).to_bytes(1, 'little'))
                f.write(int(0).to_bytes(1, 'little'))


for t in range(tests_num):
    generate_data()

    data, args = read_data()

    data = process_data(data, *args)

    write_data(data, *args)

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

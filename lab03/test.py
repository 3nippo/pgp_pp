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
    # w = np.random.randint(0, 4 * 10**8 // 10000)
    # h = 4*10**8 // w

    w = 71
    h = 71

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
            npj = w*h
            # npj = np.random.randint(0, min(w*h, 2**19) + 1)

            choosed_pixel_positions = []

            for _ in range(npj):
                pixel_position = (np.random.randint(0, h), np.random.randint(0, w))

                choosed_pixel_positions.append(pixel_position)

            single_coordinates = []

            for pixel_position in choosed_pixel_positions:
                single_coordinates.extend(pixel_position)

            sample_line = " ".join([str(npj)] + list(map(str, single_coordinates)))

            print(sample_line, file=f)


generate_data()

# for t in range(tests_num):
#     if not dont_gen_data:
#         generate_data()

#     if gen_data:
#         break

#     args = read_data()

#     result = process_data(*args)

#     write_data(*result)
#     break
#     # run C++ solution
#     process = subprocess.Popen(
#         cmd.split(),
#         stderr=subprocess.PIPE
#     )

#     _, error = process.communicate()

#     if process.returncode != 0 or error:
#         report_error(t, process.returncode, error)

#     # cmp C++ solution with Python solution
#     cmp_cmd = "cmp {} {}".format(output_name, python_output_name)

#     process = subprocess.Popen(
#         cmd.split(),
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE
#     )

#     output, error = process.communicate()

#     if process.returncode != 0 or error or output:
#         report_error(t, process.returncode, error)

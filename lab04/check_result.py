import subprocess
import sys
import numpy as np

zero = 1e-7

for testname in sys.argv[1:]:
    process = subprocess.Popen(
        "./lab04cpu".split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    with open(f"{testname}.in") as f:
        content = f.read()

    stdout, _ = process.communicate(input=content.encode())

    matrix = []

    for line in stdout.decode().split('\n'):
        row = list(map(float, line.split()))

        matrix.append(row)

    matrix.pop()

    matrix = np.array(matrix)

    input_matrix = []
    output_matrix = []

    with open(f"{testname}.in") as f:
        it = iter(f)

        n, m, k = map(int, next(it).split())

        def read_matrix(m):
            for _ in range(n):
                line = next(it)

                row = list(map(float, line.split()))

                m.append(row)

        read_matrix(input_matrix)
        read_matrix(output_matrix)

    input_matrix = np.array(input_matrix)
    output_matrix = np.array(output_matrix)

    result = input_matrix @ matrix - output_matrix

    passed = True

    good_rows = []
    for i in range(n):
        counter = 0

        for j in range(k):
            if abs(result[i, j]) > zero:
                passed = False
            else:
                counter += 1

        if counter == k:
            good_rows.append(i)

    if not passed:
        print(np.sum(np.abs(result)))

    if passed:
        print(f"{testname}: PASSED")
    else:
        print(f"{testname}: FAILED")

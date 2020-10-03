import numpy as np
import sys
import subprocess

n = 2**25 - 1

tests_num = 50

cmd = "./lab01"

for t in range(tests_num):
    if t == tests_num - 1:
        N = n
    else:
        N = np.random.randint(0, n + 1)

    N = 10

    a = np.random.randint(-n, n, N) * np.random.uniform(0, 1, N)
    b = np.random.randint(-n, n, N) * np.random.uniform(0, 1, N)

    a_str = " ".join(map(lambda x: "%.10e" % x, a))
    b_str = " ".join(map(lambda x: "%.10e" % x, b))

    process = subprocess.Popen(
        cmd.split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stdout
    )

    output, error = process.communicate(input="\n".join([str(N), a_str, b_str]).encode())

    if error:
        print(error)

    output = output.decode().rstrip()

    result = [max(a, b) for a, b in zip(a, b)]

    result_str = " ".join(map(lambda x: "%.10e" % x, result))

    passed = output == result_str

    if not passed:
        print("Test #{}:".format(t+1))
        print(a_str)
        print(b_str)
        print()

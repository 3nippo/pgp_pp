import numpy as np
import sys

size = int(sys.argv[1])

with open("t5.in", 'w') as f:
    print(f"{size} " * 3, file=f)

    counter = 0

    for _ in range(2 * size):
        row = []

        for _ in range(size):
            counter += 1
            row.append(str(counter))

        # for val in np.random.random(size):
        #     sign = np.random.randint(2)

        #     number = val * size

        #     if sign:
        #         number = -number

        #     row.append(str(number))

        print(" ".join(row), file=f)

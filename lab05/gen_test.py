import sys
import random

name, size = sys.argv[1:]

size = int(size)

with open(name, 'wb') as f:
    f.write(size.to_bytes(4, 'little'))

    for _ in range(size):
        number = random.randint(0, 10000)

        f.write(number.to_bytes(4, 'little'))

import math

with open('binary_dump', 'rb') as f:
    w = f.read(4)
    h = f.read(4)

    w, h = map(lambda x: int.from_bytes(x, 'little'), [w, h])

    m = []

    for i in range(3):
        for j in range(3):
            r = f.read(1)
            g = f.read(1)
            b = f.read(1)
            a = f.read(1)

            r, g, b = map(lambda x: int.from_bytes(x, 'little'), [r, g, b])

            y = 0.299 * r + 0.587 * g + 0.114 * b
            # u = -0.168736 * r + -0.331264 * g + 0.5 * b + 128
            # v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128

            m.append(y)

m = [m[i:i+3] for i in range(0, 9, 3)]
m = [row + [row[-1]] for row in m]
m = m + [m[-1]]


def process(m):
    new_m = []

    for i in range(3):
        for j in range(3):
            Gx = m[i][j] - m[i+1][j+1]
            Gy = m[i][j+1] - m[i+1][j]
            G = math.sqrt(Gx*Gx + Gy*Gy)
            # G = abs(Gx) + abs(Gy)
            new_m.append(G)

    return [new_m[i:i+3] for i in range(0, 9, 3)]


m = process(m)
# mr, mg, mb = map(process, [mr, mg, mb])

with open('binary_lump', 'wb') as f:
    f.write(int(3).to_bytes(4, 'little'))
    f.write(int(3).to_bytes(4, 'little'))
    f.write(int(0).to_bytes(4, 'little'))

    for i in range(3):
        for j in range(3):
            r = m[i][j]
            g = m[i][j]
            b = m[i][j]

            # r = y + 1.402 * (v - 128)
            # g = y - 0.344136 * (u - 128) - 0.714136 * (v - 128)
            # b = y + 1.772 * (u - 128)

            print(r)

            f.write(int(r).to_bytes(1, 'little'))
            f.write(int(g).to_bytes(1, 'little'))
            f.write(int(b).to_bytes(1, 'little'))
            f.write(int(0).to_bytes(1, 'little'))

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys


def from_image_to_data(imagepath, datapath):
    image = Image.open(imagepath)

    arr = np.asarray(image)

    h, w, _ = arr.shape

    with open(datapath, 'wb') as f:
        f.write(int(w).to_bytes(4, 'little'))
        f.write(int(h).to_bytes(4, 'little'))

        for i in range(h):
            for j in range(w):
                r, g, b = arr[i][j]

                r, g, b = map(int, [r, g, b])

                f.write(r.to_bytes(1, 'little'))
                f.write(g.to_bytes(1, 'little'))
                f.write(b.to_bytes(1, 'little'))
                # alpha channel
                f.write(int(0).to_bytes(1, 'little'))


def from_data_to_image(datapath, imagepath):
    data = []

    with open(datapath, 'rb') as f:
        w = f.read(4)
        h = f.read(4)

        w, h = map(lambda x: int.from_bytes(x, 'little'), [w, h])

        for _ in range(w * h):
            r = int.from_bytes(f.read(1), 'little')
            g = int.from_bytes(f.read(1), 'little')
            b = int.from_bytes(f.read(1), 'little')
            # we dont use a but have to read it
            a = int.from_bytes(f.read(1), 'little')

            data.extend([r, g, b])

    data = np.array(data, dtype=np.uint8).reshape((h, w, 3))

    image = Image.fromarray(data).convert('RGB')

    image.save(imagepath)


def from_image_to_pdf(imagepath, pdfpath):
    image = Image.open(imagepath)

    image.save(pdfpath)


def main(argv):
    if len(argv) == 1:
        print("Bad input")
        return

    class MyDict(dict):
        def __missing__(self, key):
            return "Bad input"

    mapper = MyDict({
        "to_data": from_image_to_data,
        "to_image": from_data_to_image,
        "to_pdf": from_image_to_pdf
    })

    func = mapper[argv[1]]

    if not callable(func):
        print(func)

    func(argv[2], argv[3])


if __name__ == '__main__':
    main(sys.argv)

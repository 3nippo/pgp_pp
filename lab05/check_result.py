import sys

with open(sys.argv[1], 'rb') as f:
    last_number = f.read(4)
    last_number = int.from_bytes(last_number, 'little')

    passed = True

    while (number := f.read(4)):
        number = int.from_bytes(number, 'little')

        if number < last_number:
            passed = False
            break

        last_number = number

    if passed:
        print('PASSED')
    else:
        print('FAILED')

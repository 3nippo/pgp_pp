import sys

for input_name in sys.argv[1:]:
    with open(input_name, 'r') as f:
        lines = f.readlines()

    input_name = input_name.split('.')

    input_name = input_name[:-1] + ['public'] + [input_name[-1]]

    output_name = ".".join(input_name)

    with open(output_name, 'w') as f:
        for line in lines:
            if 'private:' in line:
                line = line.replace('private:', 'public:')
            f.write(line)

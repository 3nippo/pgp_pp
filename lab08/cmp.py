import sys

nums = []

for filename in sys.argv[1:]:
    file_nums = []

    with open(filename) as f:
        for line in f:
            line_nums = list(map(float, line.split()))

            file_nums.extend(line_nums)

        nums.append(file_nums)

sys.exit(not nums[0] == nums[1])

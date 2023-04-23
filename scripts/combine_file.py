Z = []

with open('data/RTX2080Ti/va_1_240_mm_1_15') as file1:
    with open('data/va_1_240_mm_15_61') as file2:
        for line1, line2 in zip(file1, file2):
            Z.append(line1.split()+line2.split())

with open('data/va_1_240_mm_1_61', 'w') as output:
    for line in Z:
        for elem in line:
            output.write(f'{elem} ')
        output.write('\n')

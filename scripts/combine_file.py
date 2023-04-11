Z = []

with open('data/mt_8_513_sp_1_2') as file1:
    with open('data/mt_8_513_sp_2_25') as file2:
        for line1, line2 in zip(file1, file2):
            Z.append(line1.split()+line2.split())

with open('data/mt_8_513_sp_1_25', 'w') as output:
    for line in Z:
        for elem in line:
            output.write(f'{elem} ')
        output.write('\n')

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery-nogrid')

input_path = 'data/mm_1_49_mt_1_97'

Z = list()

with open(input_path) as input:
    for line in input:
        Z.append([float(time) for time in line.split()][:])

# Z = Z[16:]

opt_time = None
x, y = None, None
for i, line in enumerate(Z):
    for j, elem in enumerate(line):
        if opt_time is None or elem < opt_time:
            opt_time = elem
            x, y = i, j

print(f'optimal: {x} {y}: {opt_time}')

# plot
nslice = 1
slice_len = len(Z)//nslice
for i in range(nslice):
    fig, ax = plt.subplots()
    ax.imshow(Z[slice_len*i:min([slice_len*(i+1), len(Z)])], aspect='auto')
    plt.show()

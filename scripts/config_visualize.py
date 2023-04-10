import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery-nogrid')

input_path = 'data/comprehensive_tune_config'

Z = list()

with open(input_path) as input:
    for line in input:
        Z.append([float(time) for time in line.split()][:])

# plot
nslice = 1
slice_len = len(Z)//nslice
for i in range(nslice):
    fig, ax = plt.subplots()
    ax.imshow(Z[slice_len*i:min([slice_len*(i+1), len(Z)])])
    plt.show()

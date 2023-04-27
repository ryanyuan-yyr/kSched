import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.style.use('_mpl-gallery-nogrid')

input_path = r'data\MX250_with_temp_detection_wrong_warmup\mm_1_49_sp_1_25'


def reverse_2_tuple(t):
    x, y = t
    return (y, x)

Z = list()

with open(input_path) as input:
    for line in input:
        Z.append([float(time) for time in line.split()])#[4:61])

# Z = Z[4:61]

opt_time = None
x, y = None, None

max_time = None
z, w = None, None

for i, line in enumerate(Z):
    for j, elem in enumerate(line):
        if opt_time is None or elem < opt_time:
            opt_time = elem
            x, y = i, j

        if max_time is None or elem > max_time:
            max_time = elem
            z, w = i, j

print(f'optimal: {x} {y}: {opt_time}')
print(f'slowest: {z} {w}: {max_time}')

# plot
nslice = 1
slice_len = len(Z)//nslice
for i in range(nslice):
    fig, ax = plt.subplots()
    ax.imshow(Z[slice_len*i:min([slice_len*(i+1), len(Z)])], aspect='equal')

    # [OBSOLETE] Showing searching trace "paper\figures\searching_trace\[OBSOLETE] searching_trace_va_mm.png"
    # delta = 0.2
    # for start, end in [
    #     ((4, 1), (4, 9)),
    #     ((4, 9), (4, 17-delta)),
    #     ((4, 17-delta), (2, 17-delta)),
    #     ((2, 17+delta), (3, 17+delta)),
    #     ((3, 17+delta), (3, 19)),
    #     ((3, 19), (3, 20))
    # ]:
    #     arrow = mpatches.FancyArrowPatch(reverse_2_tuple(start), reverse_2_tuple(end),
    #                                 mutation_scale=13, arrowstyle='->')
    #     ax.add_patch(arrow)


    plt.show()

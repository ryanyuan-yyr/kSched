import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

data = json.load(open('data/searching_results.json'))['data']


def perf_eval_diff_dev(data):
    combs = [['va', 'mm'], ['va', 'sp'], ['va', 'mt'], ['sp', 'mm'], ['sp', 'mt'],
            ['mm', 'mt']]
    labels = ["{} & {}".format(*comb) for comb in combs]
    # key: device, value: list of execution time ratios in the order of combs
    exec_time_data = defaultdict(list)

    for device, device_data in data.items():
        for i, comb in enumerate(device_data):
            assert (combs[i] == comb['comb'])
            exec_time = comb['exec_time']
            exec_time_data[device].append(exec_time[2] / exec_time[0])
    elem_sum, elem_num = 0, 0
    for device, exec_time in exec_time_data.items():
        for elem in exec_time:
            elem_sum += elem
            elem_num += 1
    print('perf_eval_diff_dev {}'.format(elem_sum / elem_num))
    total_width = 0.6
    per_bar_width = total_width / len(exec_time_data)

    x_start = np.arange(len(labels)) - total_width / 2

    fig, ax = plt.subplots()

    rects = []
    for i, (device, exec_time) in enumerate(exec_time_data.items()):
        rects.append(
            ax.bar(x_start + i * per_bar_width,
                exec_time,
                per_bar_width,
                label=device))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalized performance')
    # ax.set_title('Performance evaluation of co-scheduling on different devices')
    ax.set_xticks(x_start + (total_width - per_bar_width) / 2, labels)
    ax.legend()

    for rect in rects:
        ax.bar_label(rect, padding=3, fmt='%.2f', rotation=90)

    plt.axhline(y=1, linestyle='--', color='black')
    plt.ylim(bottom=0.7)

    fig.tight_layout()

    fig.set_figwidth(16/2)
    fig.set_figheight(9/2)

    plt.show()


def perf_eval_compared_to_nonsampling(data):
    combs = [['va', 'mm'], ['va', 'sp'], ['va', 'mt'], ['sp', 'mm'],
             ['sp', 'mt'], ['mm', 'mt']]
    labels = ["{} & {}".format(*comb) for comb in combs]
    # key: device, value: list of execution time ratios in the order of combs
    exec_time_data = defaultdict(list)

    for device, device_data in data.items():
        for i, comb in enumerate(device_data):
            assert (combs[i] == comb['comb'])
            exec_time = comb['exec_time']
            exec_time_data[device].append(exec_time[1] / exec_time[0])

    elem_sum, elem_num = 0, 0
    for device, exec_time in exec_time_data.items():
        for elem in exec_time:
            elem_sum += elem
            elem_num += 1
    print('perf_eval_compared_to_nonsampling {}'.format(elem_sum / elem_num))

    total_width = 0.6
    per_bar_width = total_width / len(exec_time_data)

    x_start = np.arange(len(labels)) - total_width / 2

    fig, ax = plt.subplots()

    rects = []
    for i, (device, exec_time) in enumerate(exec_time_data.items()):
        rects.append(
            ax.bar(x_start + i * per_bar_width,
                   exec_time,
                   per_bar_width,
                   label=device))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalized performance')
    # ax.set_title('Performance evaluation of co-scheduling on different devices')
    ax.set_xticks(x_start + (total_width - per_bar_width) / 2, labels)
    ax.legend(loc='lower left')

    for rect in rects:
        ax.bar_label(rect, padding=3, fmt='%.2f', rotation=90)

    plt.axhline(y=1, linestyle='--', color='black')
    plt.ylim(bottom=0.7)

    fig.tight_layout()

    fig.set_figwidth(16 / 2)
    fig.set_figheight(9 / 2)

    plt.show()


def perf_eval_compared_to_opt(data):
    combs = [['va', 'mm'], ['va', 'sp'], ['va', 'mt'], ['sp', 'mm'],
             ['sp', 'mt'], ['mm', 'mt']]
    labels = ["{} & {}".format(*comb) for comb in combs]
    # key: device, value: list of execution time ratios in the order of combs
    exec_time_data = defaultdict(list)

    device = 'RTX2080Ti'
    for i, comb in enumerate(data[device]):
        assert (combs[i] == comb['comb'])
        exec_time = comb['exec_time']
        exec_time_data[device].append(exec_time[3] / exec_time[0])

    print(sum(exec_time_data[device]) / len(exec_time_data[device]))

    total_width = 0.15
    per_bar_width = total_width / len(exec_time_data)

    x_start = np.arange(len(labels)) - total_width / 2

    fig, ax = plt.subplots()

    rects = []
    for i, (device, exec_time) in enumerate(exec_time_data.items()):
        rects.append(
            ax.bar(x_start + i * per_bar_width,
                   exec_time,
                   per_bar_width,
                   label=device))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalized performance')
    # ax.set_title(
    #     'Performance evaluation of co-scheduling on RTX2080Ti compared to optimal scheduling')
    ax.set_xticks(x_start + (total_width - per_bar_width) / 2, labels)
    ax.legend()

    for rect in rects:
        ax.bar_label(rect, padding=3, fmt='%.2f')

    plt.axhline(y=1, linestyle='--', color='black')
    plt.ylim(bottom=0)

    fig.tight_layout()
    fig.set_figwidth(16/2)
    fig.set_figheight(9/2)
    plt.ylim(top=1.25)

    plt.show()

perf_eval_diff_dev(data)
perf_eval_compared_to_nonsampling(data)
perf_eval_compared_to_opt(data)
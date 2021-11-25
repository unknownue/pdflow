import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import csv
import math
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline
from tsmoothie.smoother import *
import matplotlib


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def y_converter(in_y):
   # if in_y < 3.1:
    #     return 2 * (in_y - 2.8) / 0.3
    # elif in_y < 3.3:
    #     return 2 * (in_y - 3.1) / 0.2 + 2
    # elif in_y < 3.9:
    #     return 2 * (in_y - 3.3) / 0.6 + 4
    # elif in_y < 20:
    #     return 2 * (in_y - 3.9) / 16.1 + 6
    # elif in_y < 60:
    #     return 2 * (in_y - 20) / 40 + 8
    # else:
    #     return 10

    # if in_y >= 90:
    #     return 1
    # if in_y < 3.1:
    #     return 0.297 * (in_y - 2.8) / 0.3
    #
    # c = (in_y - 2.8) / (90 - 2.8)
    # s = -2 / math.log(c)
    # z = -math.e ** -s + 1

    if in_y < 3.3:
        z = (in_y - 2.78) * 0.75
    else:
        z = 0.3125 * math.log(in_y - 1.2, 10)
        z = z + 0.42
        z = 0.39 + (z - 0.52) / 0.48 * 0.61
    return z


def main():
    matplotlib.use('TkAgg')
    data = pd.read_csv("eval/plot/curve-aug32.csv")
    # fig = plt.figure()
    # ax = plt.gca()
    # plt.yscale("log")
    plt.figure(figsize=(10, 7.5), dpi=80)
    plt.grid()
    plt.ylim(0, 1)

    y_raw = np.linspace(0, 1, 9)
    # y_label = [2.85, 3.05, 3.2, 3.3, 7.4, 18, 36, 73, '']
    # y_label = [2.85, 3.0, 3.15, 3.3, 7.4, 18, 36, 73, '']
    y_label = None
    plt.xlabel('Training epoch', family='Arial', fontsize=16)
    plt.ylabel('Validation loss (Chamfer distance)', family='Arial', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(y_raw, y_label, fontsize=12)
    x = np.linspace(0, 100, 101)

    # here to modify
    strs = ['cut1aug32', 'cut4aug32', 'cut8aug32', 'cut16aug32', 'cut32aug32', 'cut33aug32']
    labels = ['mask = 1', 'mask = 4', 'mask = 8', 'mask = 16', 'mask = 32', 'mask = 33']
    color_idx = [9, 1, 2, 4, 0, 5, 3]
    win_len = 6
    cmap = plt.get_cmap("tab10")



    for idx, s in enumerate(strs):
        y = np.array(list(map(y_converter, data[s])))
        smoother = ConvolutionSmoother(window_len=win_len, window_type='ones')
        smoother.smooth(y)
        plt.plot(smoother.smooth_data[0], color=cmap.colors[color_idx[idx]])


    plt.legend(labels, fontsize=12, loc='upper right')
    # plt.axhline(0.375, linestyle='--', c='black')

    plt.show()


if __name__ == "__main__":
    main()

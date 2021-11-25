
import os
import sys
import argparse
import numpy as np
import matplotlib.cm as cm
import pyvista as pv
import torch

sys.path.append(os.getcwd())
# from modules.utils.score_utils import farthest_point_sampling
from modules.utils.fps import farthest_point_sampling as torch_fps, index_points as torch_pindex



def plotting(args):

    pc = np.loadtxt(args.input_xyz, dtype=np.float32)  # [N, 3]
    pc = torch.from_numpy(pc)
    # pc, _ = farthest_point_sampling(torch.unsqueeze(pc, 0), 1024)

    idx_fps_seed = torch_fps(torch.unsqueeze(pc, 0), 256)
    pc = torch_pindex(torch.unsqueeze(pc, 0), idx_fps_seed)

    pc = torch.squeeze(pc).numpy()
    N, _ = pc.shape

    plotter = pv.Plotter()
    plotter.set_background('white')

    num_outliers = 20
    colors = torch.tensor([0, 59, 243]).repeat(N, 1)

    if num_outliers > 0:
        outlier_idx = torch.randperm(N)[:num_outliers].long()
        # print(outlier_idx, outlier_idx)
        outlier_colors = torch.tensor([252, 251, 67]).repeat(num_outliers, 1)
        colors[outlier_idx] = outlier_colors

    plotter.add_mesh(pv.PolyData(pc), scalars=colors, style='points', point_size=6, cmap=None, rgb=True)
    plotter.show(window_size=(500, 500))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_xyz', type=str, required=True, help='Path to input xyz file')
    args = parser.parse_args()

    plotting(args)

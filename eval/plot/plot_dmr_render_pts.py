
# Usage: python plot_dmr_render_pts.py --input_xyz=path/to/input.xyz --gt_xyz=path/to/gt.xyz --output_path=path/to/output.png

import os
import sys
import argparse
import numpy as np

from pathlib import Path
from sklearn.neighbors import NearestNeighbors

import matplotlib.cm as cm


BLENDER_PATH = Path('/workspace/Denoise/deflow/eval/viz_blender/blender/blender')
BLEND_FILE_PATH = Path('/workspace/Denoise/deflow/eval/viz_blender/point_clouds.blend')
RENDER_PY_PATH = Path('/workspace/Denoise/deflow/eval/viz_blender/render_one_pc.py')

RGB_XYZ_TEM_PATH = Path('/workspace/Denoise/deflow/eval/viz_blender/rendering.xyz')


def point2surface_distance(normals, pred, gt):
    """
    pred: [N1, 3]
    gt  : [N2, 3]
    """

    nbrs = NearestNeighbors(n_neighbors=16, algorithm='ball_tree').fit(gt)
    _, indices = nbrs.kneighbors(pred)  # [N1, k]
    indices = indices.flatten()  # [N1 * k,]

    n = normals[indices].reshape(-1, 16, 3)  # [N1, k, 3]

    p1 = pred.reshape(-1, 1, 3)  # [N1, 1, 3]
    p2 = gt[indices].reshape(-1, 16, 3)   # [N1, k, 3]

    numerator1 = np.sum(p1 * n, axis=-1)   # [N1, k]
    numerator2 = np.sum(p2 * n, axis=-1)   # [N1, k]

    numerator   = np.abs(numerator1 - numerator2)   # [N1, k]
    denominator = np.sqrt(np.sum(n ** 2, axis=-1))  # [N1, k]

    p2s = numerator / denominator  # [N1, k]
    p2s = np.amin(p2s, axis=-1)  # [N1,]
    return p2s


def render(xyzrgb_path, output_path, rotate_x, rotate_y, rotate_z):
    # render_cmd = '''%s --background %s --python %s -- %s %s > /dev/null''' % (BLENDER_PATH, BLEND_FILE_PATH, RENDER_PY_PATH, xyzrgb_path, output_path)
    render_cmd = '''%s --background %s --python %s -- %s %s %s %s %s''' % (BLENDER_PATH, BLEND_FILE_PATH, RENDER_PY_PATH, xyzrgb_path, output_path, rotate_x, rotate_y, rotate_z)
    os.system(render_cmd)

def colorize(p2s, cmap_name='Wistia', vmin=0, vmax=1):
    # norm = plt.Normalize(vmin, vmax)
    norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG

    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m.to_rgba(p2s)[:, :3]


def plotting(args):

    _, file_name = os.path.split(args.input_xyz)
    gt_dir, _ = os.path.split(args.gt_xyz)
    path_normal = Path(gt_dir) / file_name.replace('.xyz', '.normal')

    normals = np.loadtxt(path_normal, dtype=np.float32)
    pred = np.loadtxt(args.input_xyz, dtype=np.float32)  # [N, 3]
    gt = np.loadtxt(args.gt_xyz, dtype=np.float32)

    # calculate p2s distance for each point
    p2s = point2surface_distance(normals, pred, gt)  # [N,]

    # calculate a rgb color value for each point (rgb in [0.0, 1.0])
    rgb = colorize(p2s, cmap_name='coolwarm', vmin=0.0, vmax=0.1)
    # print(rgb.shape)
    # print(rgb[:10])

    xyzrgb = np.concatenate([pred, rgb], axis=1)
    np.savetxt(RGB_XYZ_TEM_PATH, xyzrgb, fmt='%.6f')
    render(RGB_XYZ_TEM_PATH, args.output_path, args.rotate_x, args.rotate_y, args.rotate_z)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_xyz', type=str, required=True, help='Path to input xyz file')
    parser.add_argument('--gt_xyz', type=str, required=True, help='Path to ground truth xyz file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output file')
    parser.add_argument('--rotate_x', type=int, default=0, help='Rotation along x-axis (in degrees)')
    parser.add_argument('--rotate_y', type=int, default=0, help='Rotation along y-axis (in degrees)')
    parser.add_argument('--rotate_z', type=int, default=0, help='Rotation along z-axis (in degrees)')
    args = parser.parse_args()

    plotting(args)

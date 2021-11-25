
# Usage: python plot_score_render_pts.py --input_xyz=path/to/input.xyz --gt_xyz=path/to/gt.xyz --output_path=path/to/output.png

import os
import sys
import argparse
import numpy as np
import point_cloud_utils as pcu
import matplotlib.cm as cm
# import torch

from pathlib import Path
from matplotlib import colors

BLENDER_PATH = Path('/workspace/Denoise/deflow/eval/viz_blender/blender/blender')
RENDER_PY_PATH = Path('/workspace/Denoise/deflow/eval/viz_blender/render_one_pc_fix_color.py')

BLEND_FILE_PATH = '/workspace/Denoise/deflow/eval/viz_blender/fix_point_clouds%d.blend'


def render(xyz_path, output_path, rotate_x, rotate_y, rotate_z, blend_idx=1):

    blend_file_path = BLEND_FILE_PATH % blend_idx

    # render_cmd = '''%s --background %s --python %s -- %s %s > /dev/null''' % (BLENDER_PATH, blend_file_path, RENDER_PY_PATH, xyz_path, output_path)
    render_cmd = '''%s --background %s --python %s -- %s %s %s %s %s''' % (BLENDER_PATH, blend_file_path, RENDER_PY_PATH, xyz_path, output_path, rotate_x, rotate_y, rotate_z)
    os.system(render_cmd)

def plotting(args):
    render(args.input_xyz, args.output_path, args.rotate_x, args.rotate_y, args.rotate_z, args.blend_idx)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_xyz', type=str, required=True, help='Path to input xyz file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output file')
    parser.add_argument('--rotate_x', type=int, default=0, help='Rotation along x-axis (in degrees)')
    parser.add_argument('--rotate_y', type=int, default=0, help='Rotation along y-axis (in degrees)')
    parser.add_argument('--rotate_z', type=int, default=0, help='Rotation along z-axis (in degrees)')
    parser.add_argument('--blend_idx', type=int, default=1)
    args = parser.parse_args()

    plotting(args)

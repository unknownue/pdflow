
# Usage: python plot_score_render_pts.py --input_xyz=path/to/input.xyz --gt_xyz=path/to/gt.xyz --output_path=path/to/output.png

import os
import sys
import argparse
import numpy as np
import point_cloud_utils as pcu
import matplotlib.cm as cm
import torch

from pathlib import Path
from matplotlib import colors
from PIL import ImageColor

sys.path.append(os.getcwd())

from modules.utils.score_utils import pointwise_p2m_distance_normalized



BLENDER_PATH = Path('/workspace/Denoise/deflow/eval/viz_blender/blender/blender')
RENDER_PY_PATH = Path('/workspace/Denoise/deflow/eval/viz_blender/render_one_pc_ide_color.py')
RGB_XYZ_TEM_PATH = Path('/workspace/Denoise/deflow/eval/viz_blender/rendering.xyz')
BLEND_FILE_PATH = '/workspace/Denoise/deflow/eval/viz_blender/fix_point_clouds%d.blend'


def render(xyzrgb_path, output_path, rotate_x, rotate_y, rotate_z, blend_idx=1):

    blend_file_path = BLEND_FILE_PATH % blend_idx

    # render_cmd = '''%s --background %s --python %s -- %s %s > /dev/null''' % (BLENDER_PATH, blend_file_path, RENDER_PY_PATH, xyzrgb_path, output_path)
    render_cmd = '''%s --background %s --python %s -- %s %s %s %s %s''' % (BLENDER_PATH, blend_file_path, RENDER_PY_PATH, xyzrgb_path, output_path, rotate_x, rotate_y, rotate_z)
    os.system(render_cmd)

def colorize(p2s, vmin=0, vmax=1, norm='Normalize'):

    # cmap_name = 'coolwarm'
    # cmap = cm.get_cmap(cmap_name)  # PiYG

    vals = np.ones((256, 4))

    # min_color = np.array([213, 225, 230]) / 256  # np.array([118, 170, 232]) / 256
    # mid_color = np.array([153, 182, 195]) / 256  # np.array([210, 210, 210]) / 256
    # max_color = np.array([44, 52, 64]) / 256  # np.array([255, 55, 55]) / 256
    # vals[:, 0] = np.concatenate([np.linspace(min_color[0], mid_color[0], 48), np.linspace(mid_color[0], max_color[0], 208)])
    # vals[:, 1] = np.concatenate([np.linspace(min_color[1], mid_color[1], 48), np.linspace(mid_color[1], max_color[1], 208)])
    # vals[:, 2] = np.concatenate([np.linspace(min_color[2], mid_color[2], 48), np.linspace(mid_color[2], max_color[2], 208)])

    # Light blue to Dark blue
    # min_color = np.array([213, 225, 230]) / 256  # np.array([118, 170, 232]) / 256
    # max_color = np.array([44, 52, 64])    / 256  # np.array([255, 55, 55]) / 256
    # vals[:, 0] = np.linspace(min_color[0], max_color[0], 256)
    # vals[:, 1] = np.linspace(min_color[1], max_color[1], 256)
    # vals[:, 2] = np.linspace(min_color[2], max_color[2], 256)

    # orange to red
    # min_color = np.array([255, 239, 204]) / 256
    # mid_color = np.array([255, 153, 102]) / 256
    # max_color = np.array([255, 0, 0]) / 256

    # Purpose-blue
    min_color = np.array([230, 230, 255]) / 256
    mid_color = np.array([128, 128, 255]) / 256
    max_color = np.array([0, 0, 153]) / 256

    # Green
    # min_color = np.array(list(ImageColor.getcolor("#F3FAEE", "RGB"))) / 256
    # mid_color = np.array(list(ImageColor.getcolor("#3A631C", "RGB"))) / 256
    # max_color = np.array(list(ImageColor.getcolor("#1C3609", "RGB"))) / 256

    # # orange
    # min_color = np.array(list(ImageColor.getcolor("#FFE8DC", "RGB"))) / 256
    # mid_color = np.array(list(ImageColor.getcolor("#FC600A", "RGB"))) / 256
    # max_color = np.array(list(ImageColor.getcolor("#341809", "RGB"))) / 256

    # blue
    # min_color = np.array(list(ImageColor.getcolor("#E6ECFE", "RGB"))) / 256
    # mid_color = np.array(list(ImageColor.getcolor("#7195F9", "RGB"))) / 256
    # # max_color = np.array(list(ImageColor.getcolor("#0F4CF5", "RGB"))) / 256
    # max_color = np.array([44, 52, 64]) / 256

    # # blue2
    # min_color = np.array([239, 244, 254]) / 256
    # mid_color = np.array([31, 139, 235]) / 256
    # max_color = np.array([38, 88, 148]) / 256

    vals[:, 0] = np.concatenate([np.linspace(min_color[0], mid_color[0], 127), np.linspace(mid_color[0], max_color[0], 129)])
    vals[:, 1] = np.concatenate([np.linspace(min_color[1], mid_color[1], 127), np.linspace(mid_color[1], max_color[1], 129)])
    vals[:, 2] = np.concatenate([np.linspace(min_color[2], mid_color[2], 127), np.linspace(mid_color[2], max_color[2], 129)])

    cmap = colors.ListedColormap(vals)

    if norm == 'Normalize':
        norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
    elif norm == 'Power':
        norm = cm.colors.PowerNorm(2, vmin, vmax)
    elif norm == 'TwoSlope':
        norm = cm.colors.TwoSlopeNorm(vmin + (vmax - vmin) * 0.2, vmin, vmax)
    elif norm == 'Log':
        norm = cm.colors.LogNorm(vmin + 1e-4, vmax)
    else:
        assert False
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m.to_rgba(p2s)[:, :3]


def plotting(args):

    _, file_name = os.path.split(args.input_xyz)
    path_mesh = os.path.join(args.input_off, file_name.replace('.xyz', f'.{args.mesh_format}'))

    pred  = np.loadtxt(args.input_xyz, dtype=np.float32)  # [N, 3]
    noise = np.loadtxt(args.noise_xyz, dtype=np.float32)  # [N, 3]

    # calculate p2s distance for each point
    verts, faces = pcu.load_mesh_vf(path_mesh)
    verts = torch.from_numpy(verts.astype(np.float32))
    faces = torch.from_numpy(faces.astype(np.int64))

    p2s_pred  = pointwise_p2m_distance_normalized(torch.from_numpy(pred),  verts, faces)  # [N,]
    p2s_noise = pointwise_p2m_distance_normalized(torch.from_numpy(noise), verts, faces)  # [N,]
    vmax = torch.max(p2s_noise).cpu().item()

    # calculate a rgb color value for each point (rgb in [0.0, 1.0])
    rgb = colorize(p2s_pred, vmin=0.0, vmax=vmax, norm=args.norm)
    # print(rgb.shape)
    # print(rgb[:10])

    xyzrgb = np.concatenate([pred, rgb], axis=1)
    np.savetxt(RGB_XYZ_TEM_PATH, xyzrgb, fmt='%.6f')
    render(RGB_XYZ_TEM_PATH, args.output_path, args.rotate_x, args.rotate_y, args.rotate_z, args.blend_idx)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--noise_xyz', type=str, required=True, help='Path to noise xyz file')
    parser.add_argument('--input_xyz', type=str, required=True, help='Path to input xyz file')
    parser.add_argument('--input_off', type=str, required=True, help='Path to ground-truth mesh directory')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output file')
    parser.add_argument('--rotate_x', type=int, default=0, help='Rotation along x-axis (in degrees)')
    parser.add_argument('--rotate_y', type=int, default=0, help='Rotation along y-axis (in degrees)')
    parser.add_argument('--rotate_z', type=int, default=0, help='Rotation along z-axis (in degrees)')
    parser.add_argument('--mesh_format', type=str, default='off')
    parser.add_argument('--norm', type=str, default='TwoSlope')
    parser.add_argument('--blend_idx', type=int, default=1)
    args = parser.parse_args()

    plotting(args)

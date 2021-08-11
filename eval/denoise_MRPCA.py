
import os
import sys
import argparse
import numpy as np

from pathlib import Path
from tqdm import tqdm

DEN_PROGRAM_PATH = Path('/workspace/Experiment/Denoise/MRPCA/build/mrpca')
FPS_PROGRAM_PATH = Path('/workspace/Experiment/Denoise/deflow/eval/fps_points.py')


def evaluate(args, path, split):
    _, file_name = os.path.split(path)
    output_path = Path(args.output_dir) / split / file_name

    # print('Evaluating %s...'%path)
    denoise_cmd = '''%s -i %s -o %s -k %s -l %s -t %s -s %s -r %s -f %s -e %e > /dev/null''' % (DEN_PROGRAM_PATH, path, output_path, args.k, args.l, args.t, args.s, args.r, args.f, args.e)
    # print(denoise_cmd)
    os.system(denoise_cmd)

    pts = np.loadtxt(output_path, dtype=np.float32)
    pts = pts[:, :3]
    np.savetxt(output_path, pts, fmt='%.6f')

    if args.limit_num_point is not None:
        down_cmd = '''python %s --input_file=%s --output_file=%s --num_points=%s''' % (FPS_PROGRAM_PATH, output_path, output_path, args.limit_num_point)
        os.system(down_cmd)


def mp_walkFile(func, args, split, directory):

    for root, dirs, files in os.walk(directory):
        paths = [os.path.join(root, f) for f in files if '.xyz' in f]

        # Single thread processing
        for path in tqdm(paths):
            func(args, path, split)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--limit_num_point', type=int, default=None, help='Target number of output points downsampled by fps(if not set, do not employ downsample)')
    parser.add_argument('--k', type=int, default=30, help='Neighborhood size')
    parser.add_argument('--l', type=float, default=1.0, help='Base outlier sparsity parameter')
    parser.add_argument('--t', type=float, default=1e-6, help='Data fitting parameter')
    parser.add_argument('--s', type=int, default=20, help='Normal weights bandwidth (degrees)')
    parser.add_argument('--r', type=int, default=50, help='RPCA iterations')
    parser.add_argument('--f', type=int, default=5, help='fitting iterations')
    parser.add_argument('--e', type=float, default=1e-6, help='RPCA tolerance')
    parser.add_argument('--iteration', type=int, required=True, help='Number of filter iterations')
    args = parser.parse_args()

    splits = ['train_test_0.010', 'train_test_0.020', 'train_test_0.025', 'train_test_0.030']

    for i, split in enumerate(splits):
        target_path = Path(args.input_dir) / split
        print('Evaluation for %s'%target_path)
        mp_walkFile(evaluate, args, split, target_path)

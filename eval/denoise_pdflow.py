
import os
import sys
import argparse

from pathlib import Path
from tqdm import tqdm

DEN_PROGRAM_PATH = Path('/workspace/Experiment/Denoise/deflow/models/publish/denoising.py')
# DEN_PROGRAM_PATH = Path('/workspace/Experiment/Denoise/deflow/models/deflow/denoising.py')
# FPS_PROGRAM_PATH = Path('/workspace/Experiment/Denoise/deflow/eval/fps_points.py')


def evaluate(args, path, split):
    output_path = Path(args.output_dir) / split

    # print('Evaluating %s...'%path)
    denoise_cmd = '''python %s --input=%s --output=%s --ckpt=%s --upckpt=%s --limit_num_point=%s > /dev/null''' % (DEN_PROGRAM_PATH, path, output_path, args.ckpt, args.upckpt, args.limit_num_point)
    os.system(denoise_cmd)
    # print(denoise_cmd)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to denoiser checkpoint')
    parser.add_argument('--upckpt', type=str, required=True, help='Path to upsampler checkpoint')
    parser.add_argument('--limit_num_point', type=int, default=None, help='Target number of output points downsampled by fps(if not set, do not employ downsample)')
    args = parser.parse_args()

    splits = ['train_test_0.010', 'train_test_0.020', 'train_test_0.025', 'train_test_0.030']

    for i, split in enumerate(splits):
        input_path = Path(args.input_dir) / split
        print('Evaluation for %s'%input_path)
        evaluate(args, input_path, split)

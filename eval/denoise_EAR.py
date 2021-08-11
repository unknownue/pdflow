
import os
import sys
import argparse

from pathlib import Path
from tqdm import tqdm

NOR_PROGRAM_PATH = Path('/workspace/Experiment/Denoise/EAR/normal_estimation')
DEN_PROGRAM_PATH = Path('/workspace/Experiment/Denoise/EAR/EAR')
FPS_PROGRAM_PATH = Path('/workspace/Experiment/Denoise/deflow/eval/fps_points.py')
NORMAL_OUT_PATH  = Path('/workspace/Experiment/Denoise/EAR/tmp_normal.xyz')



def evaluate(args, path, split):
    _, file_name = os.path.split(path)
    output_path = Path(args.output_dir) / split / file_name

    normal_est_cmd = '''%s %s %s > /dev/null''' % (NOR_PROGRAM_PATH, path, NORMAL_OUT_PATH)
    os.system(normal_est_cmd)

    # print('Evaluating %s...'%path)
    denoise_cmd = '''%s %s %s %s %s %s > /dev/null''' % (DEN_PROGRAM_PATH, NORMAL_OUT_PATH, output_path, args.sharpness, args.sensitivity, args.iteration)
    # print(cmd)
    os.system(denoise_cmd)

    if args.limit_num_point is not None:
        down_cmd = '''python %s --input_file=%s --output_file=%s --num_points=%s''' % (FPS_PROGRAM_PATH, output_path, output_path, args.limit_num_point)
        os.system(down_cmd)

    exit()


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
    parser.add_argument('--radius', type=float, required=True)  # [0.05, 0.30, 0.25, 0.20, 0.15, 0.10]
    parser.add_argument('--sharpness', type=float, required=True)  # [30, 25, 20, 10, 5]
    parser.add_argument('--sensitivity', type=float, required=True)  # [0, 1, 2, 3, 4, 5]
    parser.add_argument('--iteration', type=int, required=True, help='Number of filter iterations')
    args = parser.parse_args()

    splits = ['train_test_0.010', 'train_test_0.020', 'train_test_0.025', 'train_test_0.030']

    for i, split in enumerate(splits):
        target_path = Path(args.input_dir) / split
        print('Evaluation for %s'%target_path)
        mp_walkFile(evaluate, args, split, target_path)

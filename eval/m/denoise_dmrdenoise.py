
import os
import sys
import argparse

from pathlib import Path
from tqdm import tqdm


DEN_PROGRAM_PATH = Path('/workspace/Experiment/Denoise/DMRDenoise/denoise.py')
FPS_PROGRAM_PATH = Path('/workspace/Experiment/Denoise/deflow/eval/fps_points.py')
# INPUT_DIR = Path('/workspace/Datasets/DMRDenoise/dataset_test/')
# OUTPUT_DIR = Path('/workspace/Experiment/Denoise/DMRDenoise/evaluation/')
# GROUND_TRUTH_DIR = Path('/workspace/Datasets/DMRDenoise/gts_full_test_50k/')
CHECKPOINT_PATH = Path('/workspace/Experiment/Denoise/DMRDenoise/pretrained/supervised/epoch=153.ckpt')



def evaluate(args, path, split):
    _, file_name = os.path.split(path)
    output_path = Path(args.output_dir) / split / file_name

    # print('Evaluating %s...'%path)
    denoise_cmd = '''python %s --input=%s --output=%s --ckpt=%s > /dev/null''' % (DEN_PROGRAM_PATH, path, output_path, CHECKPOINT_PATH)
    # print(cmd)
    os.system(denoise_cmd)

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
    args = parser.parse_args()

    splits = ['train_test_0.010', 'train_test_0.020', 'train_test_0.025', 'train_test_0.030']
    # dnames = ['input_full_test_50k_0.010', 'input_full_test_50k_0.020', 'input_full_test_50k_0.025', 'input_full_test_50k_0.030']

    for i, split in enumerate(splits):
        target_path = Path(args.input_dir) / split
        print('Evaluation for %s'%target_path)
        mp_walkFile(evaluate, args, split, target_path)

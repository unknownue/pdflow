
import os
import sys
import argparse

from pathlib import Path
from tqdm import tqdm


DEN_PROGRAM_PATH = Path('/workspace/Experiment/Denoise/TotalDenoising/Test.py')
FPS_PROGRAM_PATH = Path('/workspace/Experiment/Denoise/deflow/eval/fps_points.py')
CHECKPOINT_PATH = Path('/workspace/Experiment/Denoise/TotalDenoising/log/model.ckpt')



def evaluate(args, path, split):
    output_path = Path(args.output_dir) / split

    # print('Evaluating %s...'%path)
    denoise_cmd = '''python %s --inputFolder=%s --modelsFolder=%s --inTrainedModel=%s --numIters=%s --gaussFilter --dataset=4 > /dev/null''' % (DEN_PROGRAM_PATH, path, output_path, CHECKPOINT_PATH, args.iterations)
    # print(cmd)
    os.system(denoise_cmd)

    if args.limit_num_point is not None:
        raise NotImplementedError()
        # down_cmd = '''python %s --input_file=%s --output_file=%s --num_points=%s''' % (FPS_PROGRAM_PATH, output_path, output_path, args.limit_num_point)
        # os.system(down_cmd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--iterations', type=str, required=True, help='Path to output directory')
    parser.add_argument('--limit_num_point', type=int, default=None, help='Target number of output points downsampled by fps(if not set, do not employ downsample)')
    args = parser.parse_args()

    splits = ['train_test_0.010', 'train_test_0.020', 'train_test_0.025', 'train_test_0.030']

    for i, split in enumerate(splits):
        target_path = Path(args.input_dir) / split
        print('Evaluation for %s'%target_path)
        evaluate(args, target_path, split)


# sudo apt install libeigen3-dev
# sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
# g++ -std=c++14 jet.cpp -o jet.out

import os
import sys
import argparse

from pathlib import Path
from tqdm import tqdm

DEN_PROGRAM_PATH = Path('/workspace/Denoise/Jet/jet.out')


def evaluate(args, path, split=None):
    _, file_name = os.path.split(path)
    if split is None:
        output_path = Path(args.output_dir) / file_name
    else:
        output_path = Path(args.output_dir) / split / file_name

    # print('Evaluating %s...'%path)
    denoise_cmd = '''%s %s %s %s %s %s > /dev/null''' % (DEN_PROGRAM_PATH, path, args.d_fitting, args.d_monge, args.neighbour_size, output_path)
    # print(denoise_cmd)
    os.system(denoise_cmd)
    # exit()

def mp_walkFile(func, args, split, directory):
    for root, dirs, files in os.walk(directory):
        paths = [os.path.join(root, f) for f in files if '.xyz' in f]
        for path in tqdm(paths):
            func(args, path, split)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--d_fitting', type=int, default=4)
    parser.add_argument('--d_monge', type=int, default=4)
    parser.add_argument('--neighbour_size', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='DMRSet')
    args = parser.parse_args()

    splits = ['train_test_0.010', 'train_test_0.020', 'train_test_0.025', 'train_test_0.030']


    if args.dataset == 'DMRSet':
        for i, split in enumerate(splits):
            target_path = Path(args.input_dir) / split
            print('Evaluation for %s'%target_path)
            mp_walkFile(evaluate, args, split, target_path)
    elif args.dataset == 'ScoreSet':
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        print('Evaluation for %s'%args.input_dir)
        mp_walkFile(evaluate, args, None, args.input_dir)
    else:
        assert False, "Unknown dataset"

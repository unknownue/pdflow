
import os
import sys
import argparse

from pathlib import Path
from tqdm import tqdm


def renaming(args, path, split):

    indir, file_name = os.path.split(path)
    shapename = file_name.replace(f'_{args.iteration}', '')
    output_path = Path(indir) / shapename

    os.rename(path, output_path)

def mp_walkFile(func, args, split, directory):

    for root, dirs, files in os.walk(directory):
        paths = [os.path.join(root, f) for f in files if '.xyz' in f]

        # Single thread processing
        for path in tqdm(paths):
            func(args, path, split)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--iteration', type=int, required=True, help='Indicating i-th iteration')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')

    args = parser.parse_args()

    splits = ['train_test_0.010', 'train_test_0.020', 'train_test_0.025', 'train_test_0.030']

    for i, split in enumerate(splits):

        target_path = Path(args.input_dir)  / f'iter{args.iteration}' / splits[i]
        print('Renaming for %s'%target_path)
        mp_walkFile(renaming, args, split, target_path)

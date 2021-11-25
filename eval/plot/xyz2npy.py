
import os
import sys
import argparse
import numpy as np
# import torch

from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())


def npy2xyz(args, path):
    _, file_name = os.path.split(path)
    output_path = Path(args.output_dir) / file_name.replace('.xyz', '.xyz.npy')

    pc = np.loadtxt(path, dtype=np.float32)
    np.save(output_path, pc)

def mp_walkFile(func, args, directory):

    for root, dirs, files in os.walk(directory):
        paths = [os.path.join(root, f) for f in files if '.xyz' in f]

        for path in tqdm(paths):
            func(args, path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    mp_walkFile(npy2xyz, args, args.input_dir)
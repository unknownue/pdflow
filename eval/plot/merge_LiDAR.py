
import os
import sys
import argparse
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())

from modules.utils.score_utils import farthest_point_sampling


def npy2xyz(args, noisy_path):
    _, file_name = os.path.split(noisy_path)
    output_path = Path(args.output_dir) / file_name
    clean_path  = Path(args.clean_dir) / file_name

    clean = np.loadtxt(clean_path, dtype=np.float32)
    noisy = np.loadtxt(noisy_path, dtype=np.float32)
    pc = np.concatenate([clean, noisy], axis=0)

    if args.num_point is not None:
        pc, _ = farthest_point_sampling(torch.from_numpy(pc).unsqueeze(0), args.num_point)
        pc = pc[0].numpy()

    np.savetxt(output_path, pc, fmt='%.6f')

def mp_walkFile(func, args, directory):

    for root, dirs, files in os.walk(directory):
        paths = [os.path.join(root, f) for f in files if '.xyz' in f]

        for path in tqdm(paths):
            func(args, path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--noisy_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_point', type=int, default=None, help='Target number of point')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    mp_walkFile(npy2xyz, args, args.noisy_dir)

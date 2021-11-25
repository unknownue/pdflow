
import os
import sys
import argparse
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())

from modules.utils.score_utils import normalize_sphere

def evaluate(args, path):
    _, file_name = os.path.split(path)
    output_path = Path(args.output_dir) / file_name

    pc = np.loadtxt(path, dtype=np.float32)
    pc = torch.from_numpy(pc)
    normalize_pc, _, _ = normalize_sphere(pc, radius=1.0)

    np.savetxt(output_path, normalize_pc.numpy(), fmt='%.6f')

def mp_walkFile(func, args, directory):

    for root, dirs, files in os.walk(directory):
        paths = [os.path.join(root, f) for f in files if '.xyz' in f]

        # Single thread processing
        for path in tqdm(paths):
            func(args, path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()

    mp_walkFile(evaluate, args, args.input_dir)

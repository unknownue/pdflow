
import os
import sys
import argparse
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())

from modules.utils.score_utils import farthest_point_sampling



def evaluate(args, path):
    _, file_name = os.path.split(path)
    output_path = Path(args.output_dir) / file_name

    pc = np.loadtxt(path, dtype=np.float32)
    N, _ = pc.shape

    random_idx = np.arange(N)
    np.random.shuffle(random_idx)
    down_pc2 = pc[random_idx[:args.num_point - 100], :]

    pc = torch.from_numpy(pc).to('cuda')
    down_pc1, _ = farthest_point_sampling(pc.unsqueeze(0), 100)
    down_pc1 = down_pc1.cpu().numpy()[0]

    down_pc = np.concatenate([down_pc1, down_pc2])

    np.savetxt(output_path, down_pc, fmt='%.6f')

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
    parser.add_argument('--num_point', type=int, required=True, help='Target number of point')
    args = parser.parse_args()

    print('Evaluation for %s'%args.input_dir)
    mp_walkFile(evaluate, args, args.input_dir)

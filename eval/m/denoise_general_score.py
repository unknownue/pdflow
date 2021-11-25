
import os
import sys
import argparse

from pathlib import Path
from tqdm import tqdm

SCORE_DIR='/workspace/Denoise/score-denoise/'

def evaluate(args, path, split=None):
    _, file_name = os.path.split(path)
    if split is None:
        output_path = Path(args.output_dir) / file_name
    else:
        output_path = Path(args.output_dir) / split / file_name

    denoise_cmd = '''cd %s && python test_single.py --input_xyz=%s --output_xyz=%s --ckpt=%s> /dev/null''' % (SCORE_DIR, path, output_path, args.ckpt)
    os.system(denoise_cmd)

def mp_walkFile(func, args, split, directory):
    for root, dirs, files in os.walk(directory):
        paths = [os.path.join(root, f) for f in files if '.xyz' in f]
        for path in tqdm(paths):
            func(args, path, split)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--ckpt', type=str, default='./pretrained/ckpt.pt')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    print('Evaluation for %s'%args.input_dir)
    mp_walkFile(evaluate, args, None, args.input_dir)

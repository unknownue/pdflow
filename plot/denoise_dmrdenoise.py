
import os
import sys
from pathlib import Path

from tqdm import tqdm


PROGRAM_PATH = Path('/workspace/Experiment/Denoise/DMRDenoise/denoise.py')
INPUT_DIR = Path('/workspace/Datasets/DMRDenoise/dataset_test/')
OUTPUT_DIR = Path('/workspace/Experiment/Denoise/DMRDenoise/evaluation/')
# GROUND_TRUTH_DIR = Path('/workspace/Datasets/DMRDenoise/gts_full_test_50k/')
CHECKPOINT_PATH = Path('/workspace/Experiment/Denoise/DMRDenoise/pretrained/supervised/epoch=153.ckpt')


def evaluate(path, split):
    _, file_name = os.path.split(path)
    output_path = OUTPUT_DIR / split / file_name

    print('Evaluating %s...'%path)
    cmd = '''python %s --input=%s --output=%s --ckpt=%s > /dev/null''' % (PROGRAM_PATH, path, output_path, CHECKPOINT_PATH)
    # print(cmd)
    os.system(cmd)

def mp_walkFile(func, split, directory):

    for root, dirs, files in os.walk(directory):
        paths = [os.path.join(root, f) for f in files if '.xyz' in f]

        # Single thread processing
        for path in tqdm(paths):
            func(path, split)

if __name__ == "__main__":

    splits = ['train_test_0.010', 'train_test_0.020', 'train_test_0.025', 'train_test_0.030']
    dnames = ['input_full_test_50k_0.010', 'input_full_test_50k_0.020', 'input_full_test_50k_0.025', 'input_full_test_50k_0.030']

    for i, split in enumerate(splits):
        target_path = INPUT_DIR / dnames[i]
        print('Evaluation for %s'%target_path)
        mp_walkFile(evaluate, split, target_path)

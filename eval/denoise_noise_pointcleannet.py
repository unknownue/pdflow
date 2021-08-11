
import os
import sys
import argparse

from pathlib import Path
from tqdm import tqdm


DEN_PROGRAM_PATH = Path('/workspace/Experiment/Denoise/pointcleannet/noise_removal/eval_pcpnet.py')
CHECKPOINT_PATH = Path('/workspace/Experiment/Denoise/pointcleannet/models/denoisingModel/')


def evaluate(args, path, split):

    indir, file_name = os.path.split(path)
    outdir = Path(args.output_dir) / f'iter{args.iteration}' / split

    if f'_{args.iteration - 1}.xyz' in file_name:
        shapename = '%s_{i}'%(file_name.replace(f'_{args.iteration - 1}.xyz', ''))
    else:
        shapename = '%s_{i}'%(file_name.replace('.xyz', ''))

    # print('Evaluating %s...'%path)
    cmd_denoise = '''python %s --nrun=%d --modeldir=%s --indir=%s --outdir=%s --shapename=%s > /dev/null''' % (DEN_PROGRAM_PATH, args.iteration, CHECKPOINT_PATH, indir, outdir, shapename)
    # print(cmd)
    os.system(cmd_denoise)


    if args.iteration == 1:
        npy_path = Path(outdir) / file_name.replace('.xyz', '_0.xyz.npy')
    else:
        npy_path = Path(indir) / file_name.replace('.xyz', '.xyz.npy')

    os.remove(npy_path)
    os.remove(outdir / shapename.replace('_{i}', f'_{args.iteration - 1}.xyz'))


def mp_walkFile(func, args, split, directory):

    if args.iteration == 1:
        name_filter = '.xyz'
    else:
        name_filter = f'_{args.iteration - 1}.xyz'

    for root, dirs, files in os.walk(directory):
        paths = [os.path.join(root, f) for f in files if name_filter in f]

        # Single thread processing
        for path in tqdm(paths):
            func(args, path, split)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--iteration', type=int, required=True, help='Indicating i-th iteration')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')

    args = parser.parse_args()

    # splits = ['train_test_0.010']
    splits = ['train_test_0.010', 'train_test_0.020', 'train_test_0.025', 'train_test_0.030']
    # dnames = ['input_full_test_50k_0.010']
    # dnames = ['input_full_test_50k_0.010', 'input_full_test_50k_0.020', 'input_full_test_50k_0.025', 'input_full_test_50k_0.030']

    for i, split in enumerate(splits):

        if args.iteration == 1:
            # target_path = Path(args.input_dir) / dnames[i]
            target_path = Path(args.input_dir) / f'iter{args.iteration - 1}' / splits[i]
        else:
            target_path = Path(args.input_dir)  / f'iter{args.iteration - 1}' / splits[i]

        print('Evaluation for %s'%target_path)
        mp_walkFile(evaluate, args, split, target_path)

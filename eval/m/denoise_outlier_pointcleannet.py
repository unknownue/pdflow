
import os
import argparse
import numpy as np

from pathlib import Path

OUTLIER_REMOVE_PROGRAM_PATH = Path('/workspace/Denoise/pointcleannet/outliers_removal/eval_pcpnet.py')
TEMPORARY_DIR = Path('/workspace/Denoise/pointcleannet/outliers_removal/TEMP')
CHECKPOINT_PATH = Path('/workspace/Denoise/pointcleannet/models/outliersRemovalModel')


def evaluate(args):
    
    # Classify outliers
    denoise_cmd = '''python %s --indir=%s --outdir=%s --modeldir=%s > /dev/null''' % (OUTLIER_REMOVE_PROGRAM_PATH, args.input_dir, TEMPORARY_DIR, CHECKPOINT_PATH)
    os.system(denoise_cmd)

    # filter out points with probs greate than 0.5
    for root, dirs, files in os.walk(TEMPORARY_DIR):
        for f in files:
            pts = np.loadtxt(Path(root) / f, dtype=np.float32)  # [N, 4]
            filter_idx = np.where(pts[:, -1] < 0.5)
            pts = pts[filter_idx][:, :3]  # [_N, 3]

            filename, _ext = os.path.splitext(os.path.basename(f))
            fix_name = f'{filename.replace("outliers_value_", "")}.xyz'
            save_path = Path(args.output_dir) / fix_name
            np.savetxt(save_path, pts, fmt='%.6f')

            os.remove(Path(args.input_dir) / fix_name.replace('.xyz', '.pidx.npy'))
            os.remove(Path(args.input_dir) / fix_name.replace('.xyz', '.xyz.npy'))


def generate_filelist_onfly(directory):
    ff = open(Path(directory) / 'validationset.txt', 'w')
    for root, dirs, files in os.walk(directory):
        for f in files:
            filename, ext = os.path.splitext(os.path.basename(f))
            if ext == '.xyz':
                ff.write("%s\n" % filename)
    ff.close()

def remove_filelist_onfly(directory):
    os.remove(Path(directory) / 'validationset.txt')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')

    args = parser.parse_args()

    generate_filelist_onfly(args.input_dir)
    evaluate(args)
    remove_filelist_onfly(args.input_dir)

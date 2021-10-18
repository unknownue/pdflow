
import os
import sys
import argparse

# pip install --no-cache-dir convertcloud

from pathlib import Path
# from tqdm import tqdm

DEN_PROGRAM_PATH = Path("/home/unknownue/Workspace/Research/Experiment/Denoise/GLR/build/run_my_main_glr.sh")
# FPS_PROGRAM_PATH = Path('/workspace/Experiment/Denoise/deflow/eval/fps_points.py')
MATLAB_PATH = Path('/usr/local/MATLAB/R2016b')

GT_DATA_PATH         = Path('/workspace/Datasets/DMRDenoise/gts_full_test_50k/')
GT_PLY_TMP_PATH1     = Path('/workspace/Experiment/Denoise/GLR/build/gt-pt.ply')
GT_PLY_TMP_PATH2     = Path('/home/unknownue/Workspace/Research/Experiment/Denoise/GLR/build/gt-pt.ply')
INPUT_PLY_TMP_PATH1  = Path('/workspace/Experiment/Denoise/GLR/build/i-pt.ply')
INPUT_PLY_TMP_PATH2  = Path('/home/unknownue/Workspace/Research/Experiment/Denoise/GLR/build/i-pt.ply')
OUTPUT_PLY_TMP_PATH1 = Path('/home/unknownue/Workspace/Research/Experiment/Denoise/GLR/build/o-pt.ply')
OUTPUT_PLY_TMP_PATH2 = Path('/workspace/Experiment/Denoise/GLR/build/o-pt.ply')



def evaluate(args, path, split):
    _, file_name = os.path.split(path)
    input_path = Path(args.input_dir0) / split / file_name
    output_path = Path(args.output_dir) / split / file_name
    gt_path = GT_DATA_PATH / file_name

    # Convert from xyz to ply
    xyz2ply_cmd = '''docker exec -w /workspace/Experiment/Denoise/deflow/ expr-pdflow cvc %s %s > /dev/null''' % (input_path, INPUT_PLY_TMP_PATH1)
    # xyz2ply_cmd = '''cvc %s %s''' % (path, INPUT_PLY_TMP_PATH)
    os.system(xyz2ply_cmd)

    gt_xyz2ply_cmd = '''docker exec -w /workspace/Experiment/Denoise/deflow/ expr-pdflow cvc %s %s > /dev/null''' % (gt_path, GT_PLY_TMP_PATH1)
    # gt_xyz2ply_cmd = '''cvc %s %s''' % (gt_path, GT_PLY_TMP_PATH)
    os.system(gt_xyz2ply_cmd)

    # print('Evaluating %s...'%path)
    denoise_cmd = '''bash %s %s %s %s %s %s> /dev/null''' % (DEN_PROGRAM_PATH, MATLAB_PATH, INPUT_PLY_TMP_PATH2, GT_PLY_TMP_PATH2, OUTPUT_PLY_TMP_PATH1, args.noise_level)
    # print(cmd)
    os.system(denoise_cmd)

    # Convert from ply to xyz
    ply2xyz_cmd = '''docker exec -w /workspace/Experiment/Denoise/deflow/ expr-pdflow cvc %s %s > /dev/null''' % (OUTPUT_PLY_TMP_PATH2, output_path)
    # ply2xyz_cmd = '''cvc %s %s''' % (OUTPUT_PLY_TMP_PATH, output_path)
    os.system(ply2xyz_cmd)

    # if args.limit_num_point is not None:
    #     down_cmd = '''python %s --input_file=%s --output_file=%s --num_points=%s''' % (FPS_PROGRAM_PATH, output_path, output_path, args.limit_num_point)
    #     os.system(down_cmd)



def mp_walkFile(func, args, split, directory):

    for root, dirs, files in os.walk(directory):
        paths = [os.path.join(root, f) for f in files if f.endswith('.xyz')]

        # Single thread processing
        for i, path in enumerate(paths):
            print(f'{i}: {os.path.split(path)[1]}')
            func(args, path, split)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_dir0', type=str, required=True, help='Path to input directory')
    parser.add_argument('--input_dir1', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--limit_num_point', type=int, default=None, help='Target number of output points downsampled by fps(if not set, do not employ downsample)')
    parser.add_argument('--noise_level', type=float, default=0.04)  # [0.02, 0.03, 0.04]
    args = parser.parse_args()

    splits = ['train_test_0.010', 'train_test_0.020', 'train_test_0.025', 'train_test_0.030']
    # splits = ['train_test_0.025', 'train_test_0.030']

    for i, split in enumerate(splits):
        target_path = Path(args.input_dir1) / split
        print('Evaluation for %s'%target_path)
        mp_walkFile(evaluate, args, split, target_path)

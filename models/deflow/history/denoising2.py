
# Usage: python models/deflow/eval_deflow.py --patch_size=1024 --ckpt ./path/to/xxx.ckpt --input path/to/input.xyz --output path/to/output.xyz
# or python models/deflow/eval_deflow.py --patch_size=1024 --ckpt ./path/to/xxx.ckpt --input path/to/input/xyz/directory --output path/to/output/xyz/directory

import os
import sys
import argparse
import numpy as np
import torch

sys.path.append(os.getcwd())

from pathlib import Path
from glob import glob
from tqdm import tqdm

from models.deflow.deflow import DenoiseFlow
# from modules.utils.modules import BatchIdxIter
from modules.utils.patch import PatchHelper
# from modules.utils.hibertsort import HilbertSort3D

PATCH_DIVIDE = 'knn'  # 'split'


@torch.no_grad()
def run_denoise(pc, network, patch_size, expand_ratio, npoint=None):

    pc = torch.unsqueeze(pc, dim=0)
    patch_helper = PatchHelper(npoint_patch=patch_size, patch_expand_ratio=expand_ratio, patch_divide=PATCH_DIVIDE)
    denoised = patch_helper.denoise(denoiser=network, pc=pc, npoint=npoint)
    denoised = torch.squeeze(denoised)

    return denoised

def get_denoise_net(ckpt_path):
    network = DenoiseFlow(pc_channel=3)
    state_dict = torch.load(ckpt_path)
    network.load_state_dict(state_dict)
    network.init_as_trained_state()
    return network

def auto_denoise_file(args, ipath, opath, network=None):

    if network is None:
        network = get_denoise_net(args.ckpt).to(args.device)

    pc = np.loadtxt(ipath, dtype=np.float32)

    # if PATCH_DIVIDE == 'split':
    #     import pyHilbertSort as m
    #     pc, _idx_sort = m.hilbertSort(3, pc)
    #     pc = pc.astype(np.float32)

    pc = torch.from_numpy(pc).to(args.device)

    denoised = run_denoise(pc, network, args.patch_size, args.patch_expand_ratio, args.limit_num_point)
    denoised = denoised.cpu().numpy()
 
    np.savetxt(opath, denoised, fmt='%.6f')

def denoise_directory(args, idir, odir):
    ipaths = glob(f'{idir}/*.xyz')
    network = get_denoise_net(args.ckpt).to(args.device)

    for ipath in tqdm(ipaths):
        _, filename = os.path.split(ipath)
        opath = Path(odir) / filename
        auto_denoise_file(args, ipath, opath, network)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', type=str, default='path/to/file_or_directory', help='Input file path or input directory')
    parser.add_argument('--output', type=str, default='path/to/file_or_directory', help='Output file path or output directory')
    parser.add_argument('--ckpt', type=str, default='pretrain/deflow.ckpt', help='Path to network checkpoint')
    parser.add_argument('--patch_size', type=int, required=True, help='number of point in denoised patch')
    parser.add_argument('--patch_expand_ratio', type=int, default=3)
    parser.add_argument('--limit_num_point', type=int, default=None, help='Target number of output points downsampled by fps(if not set, do not employ downsample)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if os.path.isdir(Path(args.input)) and (not os.path.exists(Path(args.output)) or os.path.isdir(Path(args.output))):
        if not os.path.exists(Path(args.output)):
            os.path.os.mkdir(Path(args.output))
        denoise_directory(args, args.input, args.output)
    elif os.path.isfile(Path(args.input)):
        odir, _ = os.path.split(Path(args.output))
        if not os.path.exists(Path(odir)):
            os.path.os.mkdir(odir)
        auto_denoise_file(args, args.input, args.output)
    else:
        assert False, "Invalid input or output path"
    
    print(f'Finish denoising {args.input}...')

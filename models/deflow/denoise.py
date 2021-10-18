
import os
import sys
import argparse
import torch
import numpy as np

from glob import glob
from pathlib import Path
from pytorch3d.ops import knn_points

sys.path.append(os.getcwd())

from models.deflow.deflow import DenoiseFlow
from dataset.scoredenoise.transforms import NormalizeUnitSphere
from modules.utils.score_utils import farthest_point_sampling


@torch.no_grad()
def patch_denoise(network, pcl_noisy, patch_size=1000, seed_k=3):
    """
    pcl_noisy:  Input point cloud, [N, 3]
    """
    assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be [N, 3].'
    N, d = pcl_noisy.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # [1, N, 3]
    seed_pnts, _ = farthest_point_sampling(pcl_noisy, int(seed_k * N / patch_size))
    _, _, patches = knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)

    # patches = torch.squeeze(patches, dim=0)  # [N, K, 3]
    patches = patches.view(-1, patch_size, 3)
    patches_denoised, _, _ = network(patches)

    pcl_denoised, _fps_idx = farthest_point_sampling(patches_denoised.view(1, -1, d), N)
    return torch.squeeze(pcl_denoised, dim=0)

def denoise_pcl(args, ipath, opath, network=None):
    if network is None:
        network = get_denoise_net(args.ckpt).to(args.device)
 
    pcl_raw = torch.from_numpy(np.loadtxt(ipath, dtype=np.float32))
    # Normalize
    pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_raw)

    # Denoise
    pcl_next = pcl_noisy.to(args.device)
    for _ in range(args.niters):
        pcl_next = patch_denoise(network, pcl_next, args.patch_size, args.seed_k)
    pcl_denoised = pcl_next.cpu()

    # Denormalize
    pcl_denoised = pcl_denoised * scale + center
    np.savetxt(opath, pcl_denoised.numpy(), fmt='%.8f')
    
def get_denoise_net(ckpt_path):
    network = DenoiseFlow(pc_channel=3)
    network.load_state_dict(torch.load(ckpt_path))
    network.init_as_trained_state()
    network.eval()
    return network

if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--input', type=str, default='path/to/file_or_directory', help='Input file path or input directory')
    parser.add_argument('--output', type=str, default='path/to/file_or_directory', help='Output file path or output directory')
    parser.add_argument('--ckpt', type=str, default='pretrain/deflow.ckpt', help='Path to network checkpoint')
    parser.add_argument('--seed_k', type=int, default=3)
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--niters', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    path_input  = Path(args.input)
    path_output = Path(args.output)

    if os.path.isdir(path_input) and os.path.exists(path_input):
        if not os.path.exists(path_output):
            os.mkdir(path_output)
        
        network = get_denoise_net(args.ckpt).to(args.device)
        ipaths = glob(f'{path_input}/*.xyz')

        for ipath in ipaths:
            _, f = os.path.split(ipath)
            opath = path_output / f

            if args.verbose:
                print("Denoising {}", path_input)
            denoise_pcl(args, ipath, opath, network)

    elif os.path.isfile(path_input):
        if os.path.isfile(path_output):
            odir, _ = os.path.split(path_output)
        elif os.path.isdir(path_output):
            odir = path_output
            _, ofile = os.path.split(path_input)
            path_output = odir / ofile
        else:
            assert False
        
        if not os.path.exists(odir):
            os.mkdir(odir)
        
        if args.verbose:
            print("Denoising {}", path_input)
        denoise_pcl(args, path_input, path_output, None)
    else:
        assert False, "Invalid input or output path"

    if args.verbose:
        print(f'Finish denoising {args.input}...')

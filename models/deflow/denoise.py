
import os
import sys
import argparse
import torch
import math
import numpy as np

from glob import glob
from pathlib import Path
from tqdm import tqdm
from pytorch3d.ops import knn_points
from sklearn.cluster import KMeans

sys.path.append(os.getcwd())

from models.deflow.deflow import DenoiseFlow, Disentanglement, DenoiseFlowMLP
from dataset.scoredenoise.transforms import NormalizeUnitSphere
from modules.utils.score_utils import farthest_point_sampling, remove_outliers



@torch.no_grad()
def patch_denoise(network, pcl_noisy, patch_size=1000, seed_k=3, down_N=None):
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

    download_sample_N = (down_N or N) + 10
    pcl_denoised, _fps_idx = farthest_point_sampling(patches_denoised.view(1, -1, d), download_sample_N)

    if True:
        pcl_denoised = remove_outliers(pcl_denoised, pcl_noisy, num_outliers=10)
    return torch.squeeze(pcl_denoised, dim=0)

@torch.no_grad()
def large_patch_denoise_v1(network, pcl, cluster_size, seed=0, device='cpu', verbose=False):
    pcl = pcl.cpu().numpy()

    if verbose:
        print('Running KMeans to construct clusters...')
    n_clusters = math.ceil(pcl.shape[0] / cluster_size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(pcl)

    pcl_parts = []
    it = tqdm(range(n_clusters), desc='Denoise Clusters') if verbose else range(n_clusters)

    for i in it:
        pts_idx = kmeans.labels_ == i

        # pcl_part_noisy = torch.FloatTensor(pcl[pts_idx]).to(device)
        pcl_part_noisy = torch.from_numpy(pcl[pts_idx]).to(device)
        pcl_part_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_part_noisy)
        pcl_part_denoised = patch_denoise(network, pcl_part_noisy, seed_k=3)
        pcl_part_denoised = pcl_part_denoised * scale + center
        pcl_parts.append(pcl_part_denoised)

    return torch.cat(pcl_parts, dim=0)

@torch.no_grad()
def large_patch_denoise_v2(network, pcl, cluster_size, seed=0, device='cpu', verbose=False):
    pcl = pcl.cpu().numpy()

    if verbose:
        print('Running KMeans to construct clusters...')
    n_clusters = math.ceil(pcl.shape[0] / cluster_size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(pcl)

    pcl_parts = []
    it = tqdm(range(n_clusters), desc='Denoise Clusters') if verbose else range(n_clusters)
    expand_ratio = 1.0

    for i in it:
        pts_idx = kmeans.labels_ == i
        pcl_cluster = pcl[pts_idx]  # [N, 3]
        N_cluster, _ = pcl_cluster.shape

        cluster_center = kmeans.cluster_centers_[i]  # [3,]
        radius_squares = np.sum((pcl_cluster - cluster_center) ** 2, axis=-1)  # [N,]
        radius_square = np.average(np.sort(radius_squares)[-5:])  # scalar

        dist_square = np.sum((pcl - cluster_center) ** 2, axis=-1)  # [N,]
        expand_idx = dist_square < (radius_square * expand_ratio)
        final_pts_idx = np.logical_or(expand_idx, pts_idx)

        pcl_part_noisy = torch.from_numpy(pcl[final_pts_idx]).to(device)
        pcl_part_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_part_noisy)
        pcl_part_denoised = patch_denoise(network, pcl_part_noisy, seed_k=3, down_N=N_cluster)
        pcl_part_denoised = pcl_part_denoised * scale + center

        pcl_parts.append(pcl_part_denoised)

    return torch.cat(pcl_parts, dim=0)

def denoise_loop(args, network, pcl_raw, iters=None):
    # Normalize
    pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_raw)

    pcl_next = pcl_noisy.to(args.device)
    iters = iters or 1

    for _ in range(iters):
        pcl_next = patch_denoise(network, pcl_next, args.patch_size, args.seed_k)
    pcl_denoised = pcl_next.cpu()
    # Denormalize
    return pcl_denoised * scale + center

def denoise_partition_loop(args, network, pcl_raw):
    N, _ = pcl_raw.shape
    nsplit = N // 10000
    _, fps_idx = farthest_point_sampling(pcl_raw.view(1, N, 3), N)  # [N,]

    pcl_subs = []
    for i in range(nsplit):
        sub_idx = fps_idx[0][i::nsplit]
        pcl_sub_noisy = pcl_raw[sub_idx]  # [N / nsplit, 3]
        pcl_sub_denoised = denoise_loop(args, network, pcl_sub_noisy, iters=None)
        pcl_subs.append(pcl_sub_denoised)
    pcl_denoised = torch.cat(pcl_subs, dim=0)  # [N, 3]

    pcl_denoised, _fps_idx = farthest_point_sampling(pcl_denoised.view(1, N, 3), N)
    pcl_denoised = torch.squeeze(pcl_denoised, dim=0)
    return pcl_denoised

def denoise_pcl(args, ipath, opath, network=None):
    if network is None:
        network = get_denoise_net(args.ckpt).to(args.device)
 
    pcl_raw = torch.from_numpy(np.loadtxt(ipath, dtype=np.float32))

    # Denoise
    if pcl_raw.shape[0] > 50000:
        if args.verbose:
            print(f"Denoising {ipath}")
        # pcl_denoised = large_patch_denoise_v1(network, pcl_raw, args.cluster_size, args.seed, args.device, args.verbose)
        pcl_denoised = large_patch_denoise_v2(network, pcl_raw, args.cluster_size, args.seed, args.device, args.verbose)
        pcl_denoised = pcl_denoised.cpu()

    elif args.first_iter_partition and pcl_raw.shape[0] > 10000:
        for i in range(args.niters):
            if i == 0:
                pcl_next = denoise_partition_loop(args, network, pcl_raw)
            else:
                pcl_next = denoise_loop(args, network, pcl_next, iters=None)
        pcl_denoised = pcl_next
    else:  # number of points <= 50000
        pcl_denoised = denoise_loop(args, network, pcl_raw, iters=args.niters)

    np.savetxt(opath, pcl_denoised.numpy(), fmt='%.8f')

def get_denoise_net(ckpt_path):
    if Disentanglement.FBM.name in ckpt_path:
        disentangle = Disentanglement.FBM
    if Disentanglement.LBM.name in ckpt_path:
        disentangle = Disentanglement.LBM
    if Disentanglement.LCC.name in ckpt_path:
        disentangle = Disentanglement.LCC

    if 'MLP' in ckpt_path:
        network = DenoiseFlowMLP(disentangle, pc_channel=3)
    else:
        network = DenoiseFlow(disentangle, pc_channel=3)
    
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
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--cluster_size', type=int, default=10000)
    parser.add_argument('--niters', type=int, default=1)
    parser.add_argument('--first_iter_partition', action='store_true')
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

            denoise_pcl(args, ipath, opath, network)

    elif os.path.isfile(path_input):
        if '.' in args.output:
            # os.path.isfile(path_output):
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
            print(f"Denoising {path_input}")
        denoise_pcl(args, path_input, path_output, None)
    else:
        assert False, "Invalid input or output path"

    if args.verbose:
        print(f'Finish denoising {args.input}...')

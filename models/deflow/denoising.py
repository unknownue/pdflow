
# Usage: python models/deflow/eval_deflow.py --ckpt ./path/to/xxx.ckpt --input path/to/input.xyz --output path/to/output.xyz

import os
import sys
import argparse
import numpy as np
import math
import torch

sys.path.append(os.getcwd())

from pathlib import Path
from glob import glob
from tqdm import tqdm

from modules.utils.fps import normalize_point_cloud_numpy
from models.deflow.deflow import DenoiseFlow
from modules.utils.modules import BatchIdxIter

from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph


@torch.no_grad()
def run_denoise(pc, network, patch_size, device, random_state=0, expand_knn=16, verbose=False):
    pc, center, scale = normalize_point_cloud_numpy(pc)

    if verbose:
        print(f'[INFO] Center {repr(center)} | Scale {scale:%.6f}')

    n_clusters = math.ceil(pc.shape[0] / patch_size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(pc)
    knn_graph = kneighbors_graph(pc, n_neighbors=expand_knn, mode='distance', include_self=False)
    knn_idx = np.array(knn_graph.tolil().rows.tolist())

    patches = []
    extra_points = []
    for i in range(n_clusters):
        pts_idx = kmeans.labels_ == i
        expand_idx = np.unique(knn_idx[pts_idx].flatten())
        extra_idx = np.setdiff1d(expand_idx, np.where(pts_idx))

        patches.append(pc[expand_idx])
        extra_points.append(pc[extra_idx])

    # Denoise patch by patch
    denoised_patches = []
    for patch in patches:
        patch = torch.FloatTensor(patch).unsqueeze(0).to(device)
        pred, _, _ = network(patch)
        pred = pred.detach().cpu().reshape(-1, 3).numpy()
        denoised_patches.append(pred)
    denoised = np.concatenate(denoised_patches, axis=0)

    # denoised_patches = []
    # total_batch = len(patches)
    # patches = np.array(patches, dtype=np.float32)
    # for _, batch_idx in BatchIdxIter(batch_size=8, N=total_batch):
    #     sub_patches = patches[batch_idx]  # [B, N, C]
    #     sub_patches = torch.from_numpy(sub_patches).to(device)
    #     pred, _ = network(sub_patches)  # [B, N, C]
    #     pred = pred.detach().cpu().numpy()
    #     denoised_patches.append(pred)  # list of [B, N, C]
    # denoised = np.concatenate(denoised_patches, axis=0).reshape(-1, 3)

    denoised = (denoised / scale) + center

    return denoised

def run_denoise_middle_point_cloud(pc, network, num_splits, patch_size, device, random_state=0, expand_knn=16, verbose=False):
    np.random.shuffle(pc)  # shuffle points order in memory
    split_size = math.floor(pc.shape[0] / num_splits)
    splits = []

    for i in range(num_splits):
        if i < num_splits - 1:
            splits.append(pc[i * split_size : (i + 1) * split_size])
        else:
            splits.append(pc[i * split_size :])
    
    denoised = []
    for i, spl_pc in enumerate(tqdm(splits)):
        denoise = run_denoise(spl_pc, network, patch_size, device, random_state, expand_knn, verbose)
        denoised.append(denoise)
    return np.vstack(denoised)

def run_denoise_large_point_cloud(pc, network, cluster_size, patch_size, device, random_state=0, expand_knn=16, verbose=False):
    n_clusters = math.ceil(pc.shape[0] / cluster_size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(pc)

    knn_graph = kneighbors_graph(pc, n_neighbors=expand_knn, mode='distance', include_self=False)
    knn_idx = np.array(knn_graph.tolil().rows.tolist())

    centers = []
    patches = []
    for i in range(n_clusters):
        pts_idx = kmeans.labels_ == i
        raw_pc = pc[pts_idx]
        centers.append(np.mean(raw_pc, axis=0, keepdims=True))

        expand_idx = np.unique(knn_idx[pts_idx].flatten())
        patches.append(pc[expand_idx])

        if verbose:
            print('[INFO] Cluster Size: ', patches[-1].shape[0])
        
    denoised = []
    for i, patch in enumerate(tqdm(patches)):
        denoise = run_denoise(patch - centers[i], network, patch_size, device, random_state, expand_knn, verbose)
        denoise += centers[i]
        denoised.append(denoise)
    return np.vstack(denoised)


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
    N, _ = pc.shape

    if N >= 120000:
        if args.verbose:
            print('[INFO] Denoising large point cloud: %s' % ipath)
        denoised = run_denoise_large_point_cloud(pc, network, args.cluster_size, args.patch_size, args.device, args.seed, args.expand_knn, args.verbose)
    elif N >= 60000:
        if args.verbose:
            print('[INFO] Denoising middle-sized point cloud: %s' % ipath)
        denoised = run_denoise_middle_point_cloud(pc, network, args.num_splits, args.patch_size, args.device, args.seed, args.expand_knn, args.verbose)
    elif N >= 10000:
        if args.verbose:
            print('[INFO] Denoising regular-sized point cloud: %s' % ipath)
        denoised = run_denoise(pc, network, args.patch_size, args.device, args.seed, args.expand_knn, args.verbose)
    else:
        assert False, "Our pretrained model does not support point clouds with less than 10K points"
    
    np.savetxt(opath, denoised, fmt='%.6f')


def denoise_directory(args, idir, odir):
    ipaths = glob(f'{idir}/*.xyz')
    network = get_denoise_net(args.ckpt).to(args.device)

    for ipath in ipaths:
        _, filename = os.path.split(ipath)
        opath = Path(odir) / filename
        auto_denoise_file(args, ipath, opath, network)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', type=str, default='path/to/file_or_directory', help='Input file path or input directory')
    parser.add_argument('--output', type=str, default='path/to/file_or_directory', help='Output file path or output directory')
    parser.add_argument('--ckpt', type=str, default='pretrain/deflow.ckpt', help='Path to network checkpoint')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--expand_knn', type=int, default=16)
    parser.add_argument('--cluster_size', type=int, default=30000, help='Number of clusters for large point clouds')
    parser.add_argument('--num_splicts', type=int, default=2, help='Number of splits for middle-sized point clouds')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verbose', action='store_true')
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

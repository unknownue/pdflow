
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
# from models.deflow.pdeflow import ExDenoiseFlow
from modules.utils.modules import BatchIdxIter
from eval.fps_points import fps_numpy

from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

from models.puflow.nets.interpflow import PointInterpFlow
from models.puflow.utils.patch import PatchHelper as UpsamplePatchHelper


@torch.no_grad()
def run_upsample(pc, upsampler, device, npoint):
    patch_helper = UpsamplePatchHelper(npoint_patch=256, patch_expand_ratio=3, extract='knn')
    pc = torch.from_numpy(pc).unsqueeze(0).to(device)
    pc = patch_helper.upsample(upsampler, pc, npoint, upratio=4)
    # TODO: Try remove outlier
    pc = pc.squeeze().cpu().numpy()
    return pc


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

def post_processing(args, pc, upsampler):
    # Download sample
    down_pc = fps_numpy(pc, K=args.limit_num_point // 4)
    up_pc = run_upsample(down_pc, upsampler, args.device, args.limit_num_point)
    return up_pc

def get_upsample_net(ckpt_path):
    network = PointInterpFlow(pc_channel=3, num_neighbors=8)
    network.load_state_dict(torch.load(ckpt_path))
    network.set_to_initialized_state()
    return network

def get_denoise_net(ckpt_path):
    network = DenoiseFlow(pc_channel=3)
    state_dict = torch.load(ckpt_path)
    network.load_state_dict(state_dict)
    network.init_as_trained_state()
    return network


def auto_denoise_file(args, ipath, opath, denoiser=None, upsampler=None):

    if denoiser is None:
        denoiser = get_denoise_net(args.ckpt).to(args.device)
    if upsampler is None and args.upckpt is not None:
        upsampler = get_upsample_net(args.upckpt).to(args.device)

    pc = np.loadtxt(ipath, dtype=np.float32)
    N, _ = pc.shape

    if N >= 120000:
        if args.verbose:
            print('[INFO] Denoising large point cloud: %s' % ipath)
        denoised = run_denoise_large_point_cloud(pc, denoiser, args.cluster_size, args.patch_size, args.device, args.seed, args.expand_knn, args.verbose)
    elif N >= 60000:
        if args.verbose:
            print('[INFO] Denoising middle-sized point cloud: %s' % ipath)
        denoised = run_denoise_middle_point_cloud(pc, denoiser, args.num_splits, args.patch_size, args.device, args.seed, args.expand_knn, args.verbose)
    elif N >= 10000:
        if args.verbose:
            print('[INFO] Denoising regular-sized point cloud: %s' % ipath)
        denoised = run_denoise(pc, denoiser, args.patch_size, args.device, args.seed, args.expand_knn, args.verbose)
    else:
        assert False, "Our pretrained model does not support point clouds with less than 10K points"

    if args.upckpt is not None:
        denoised = post_processing(args, denoised, upsampler)
    np.savetxt(opath, denoised, fmt='%.6f')


def denoise_directory(args, idir, odir):
    ipaths = glob(f'{idir}/*.xyz')
    denoiser = get_denoise_net(args.ckpt).to(args.device)

    if args.upckpt is not None:
        upsampler = get_upsample_net(args.upckpt).to(args.device)
    else:
        upsampler = None

    for ipath in ipaths:
        _, filename = os.path.split(ipath)
        opath = Path(odir) / filename
        auto_denoise_file(args, ipath, opath, denoiser, upsampler)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', type=str, default='path/to/file_or_directory', help='Input file path or input directory')
    parser.add_argument('--output', type=str, default='path/to/file_or_directory', help='Output file path or output directory')
    parser.add_argument('--ckpt', type=str, default='pretrain/pdflow.ckpt', help='Path to denoiser network checkpoint')
    parser.add_argument('--upckpt', type=str, default=None, help='Path to upsampler network checkpoint')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--expand_knn', type=int, default=16)
    parser.add_argument('--cluster_size', type=int, default=30000, help='Number of clusters for large point clouds')
    parser.add_argument('--num_splicts', type=int, default=2, help='Number of splits for middle-sized point clouds')
    parser.add_argument('--limit_num_point', type=int, default=None, help='Target number of output points downsampled by fps(if not set, do not employ downsample)')
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


import torch
import torch.nn as nn
import warnings
import math

from torch import Tensor

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from knn_cuda import KNN

from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from modules.utils.fps import farthest_point_sampling as torch_fps, index_points as torch_pindex



# -----------------------------------------------------------------------------------------
class PatchHelper(object):

    def __init__(self, npoint_patch: int, patch_expand_ratio: float, patch_divide='knn'):
        """
        npoint_patch: number of point in each patch
        patch_expand_ratio: TODO
        extract: only support knn now
        """
        super(PatchHelper, self).__init__()

        self.__npoint_patch  = npoint_patch
        self.__patch_expand_ratio = patch_expand_ratio
        self.__patch_divide = patch_divide

        if self.__patch_divide == 'knn':
            self.knn = KNN(k=self.__npoint_patch, transpose_mode=False)

    def denoise(self, denoiser: nn.Module, pc: Tensor, npoint: int=None, **kwargs):
        """
        Upsample given point cloud in patches, and sample given number of point from it.
        params:
            denoiser: the network used to denoise point patch
            pc: the point cloud waited to be upsampled, in [B, N, 3]
            npoint: the number of output point of each point cloud
        """
        B, N, C = pc.shape
        pc, g_centroid, g_furthest_distance = PatchHelper.normalize_pc(pc)

        if self.__patch_divide == 'knn':
            patches = []
            patches = PatchHelper.extract_knn_patch(pc, self.knn, self.__npoint_patch, self.__patch_expand_ratio)  # [B, n_patch, k1, 3]
            patches = patches.flatten(0, 1)  # [B, N, 3]
            patches = patches.reshape(B, -1, self.__npoint_patch, C)  # [B, n_patch, k1, 3]
        elif self.__patch_divide == 'split':
            num_patch = math.floor(pc.shape[1] / self.__npoint_patch)
            last_idx = num_patch * self.__npoint_patch
            # print(pc.shape, num_patch, last_idx)
            patches = torch.reshape(pc[:, :last_idx], (B, num_patch, self.__npoint_patch, C))
        else:
            raise NotImplementedError()

        if patches.shape[1] < 18:
            predict_patches = PatchHelper.__denoise_patches(denoiser, patches, **kwargs)
        else:
            predict_patches = []
            start, total = 0, patches.shape[1]
            while start < total:
                end = min(start + 18, total)
                partical_patch = PatchHelper.__denoise_patches(denoiser, patches[:, start: end], **kwargs)
                predict_patches.append(partical_patch)
                start = start + 18
            predict_patches = torch.cat(predict_patches, dim=1)
        # [B, n_patch * self.__duplicate_patch, k1 * upratio, 3]
        # predict_patches = PatchHelper.__denoise_patches(denoiser, patches, **kwargs)

        predict_pc = PatchHelper.merge_patches(predict_patches, npoint)

        predict_pc = predict_pc * g_furthest_distance + g_centroid.transpose(1, 2)
        predict_pc = predict_pc.transpose(1, 2).contiguous()

        return predict_pc  # [B, npoint, 3]

    @staticmethod
    def __denoise_patches(denoiser: nn.Module, patches: Tensor, **kwargs):

        B, n_patch, k1, C = patches.shape
        patches = patches.reshape(B * n_patch, k1, C)
        patches, centroids, furthest_distance = PatchHelper.normalize_pc(patches)

        predict_patches = denoiser.denoise(patches, **kwargs)  # [B * n_patch, k2, C]
        predict_patches = predict_patches * furthest_distance + centroids

        predict_patches = predict_patches.reshape(B, n_patch, -1, C)
        return predict_patches

    @staticmethod
    def __extract_idx_patches(pc: Tensor, knn_searcher: KNN, npoint_patch: int, expand_ratio: float, seed_centroids_idx=None) -> Tensor:
        _, N, _ = pc.shape
        pc_T = pc.transpose(1, 2).contiguous()  # [B, C, N]

        if seed_centroids_idx is None:
            n_patch = int(N / npoint_patch * expand_ratio)
            patch_centroids_idx = furthest_point_sample(pc, n_patch)  # [B, n_patch]
        else:
            _, n_patch = seed_centroids_idx.shape
            patch_centroids_idx = seed_centroids_idx
        patch_centroids = gather_operation(pc_T, patch_centroids_idx)  # [B, C, n_patch]
        _, idx_patches = knn_searcher(pc_T, patch_centroids)  # [B, k, n_patch]

        return idx_patches, n_patch

    @staticmethod
    def extract_knn_patch(pc: Tensor, knn_searcher: KNN, npoint_patch: int, expand_ratio: float, seed_centroids_idx=None) -> Tensor:
        """
        Extract patches from point clouds by KNN. (Only work for 3D points)
        pc: Initail Point Cloud, in [B, N, C]
        """
        B, _, C = pc.shape
        idx_b = torch.arange(B).view(-1, 1)

        idx_patches, n_patch = PatchHelper.__extract_idx_patches(pc, knn_searcher, npoint_patch, expand_ratio, seed_centroids_idx)  # [B, k, n_patch]
        idx_patches = idx_patches.transpose(1, 2).flatten(start_dim=1)  # [B, n_patch * k]

        patches = pc[idx_b, idx_patches]  # [B, n_patch * k, C]
        return patches.reshape(B, n_patch, npoint_patch, C)  # [B, n_patch, k, C]

    @staticmethod
    def fps(pc: Tensor, n_point: int, transpose=True):
        """
        pc: [B, N, C]
        n_point: number of output point
        """
        centroids_idx = furthest_point_sample(pc, n_point)  # [B, n_patch]
        centroids = gather_operation(pc.transpose(1, 2).contiguous(), centroids_idx)  # [B, C, n_patch]

        if transpose is True:
            return centroids.transpose(1, 2).contiguous()
        else:
            return centroids

    @staticmethod
    def merge_patches(patches: Tensor, npoint: int=None):
        """
        patches: input patches, in [B, n_patch, k, 3]
        npoint: number of final points in each point cloud
        origins: Optional point cloud, in [B, N, 3]
        """
        B, _, _, C = patches.shape
        patches = patches.reshape(B, -1, C).contiguous()  # [B, n_patch * k, 3]
        patches_T = patches.transpose(1, 2).contiguous()

        if npoint is None:
            return patches_T
        else:
            idx_predict_pc = furthest_point_sample(patches, npoint)
            final_pc = gather_operation(patches_T, idx_predict_pc)
            return final_pc  # [B, 3, npoint]

    @staticmethod
    def merge_pc(pc1: Tensor, pc2: Tensor, npoint: int):
        tmp = torch.cat([pc1, pc2], dim=1)
        idx_fps_seed = torch_fps(tmp, npoint)
        return torch_pindex(tmp, idx_fps_seed)

    @staticmethod
    def normalize_pc(pc: Tensor):
        """
        Normalized point cloud in range [-1, 1].
        pc: [B, N, 3]
        """
        centroid = torch.mean(pc, dim=1, keepdim=True)  # [B, 1, 3]
        pc = pc - centroid  # [B, N, 3]
        dist_square = torch.sum(pc ** 2, dim=-1, keepdim=True).sqrt()  # [B, N, 1]
        furthest_distance, _ = torch.max(dist_square, dim=1, keepdim=True)  # [B, 1, 1]
        pc = pc / furthest_distance
        return pc, centroid, furthest_distance

    # @staticmethod
    # def jitter_perturbation_point_cloud(pc, sigma=0.010, clip=0.020):
    #     """
    #     Randomly jitter points. jittering is per point.
    #     Input : Original point clouds, in [B, N, 3]
    #     Return: Jittered point clouds, in [B, N, 3]
    #     """
    #     if sigma > 0:
    #         B, N, C = pc.shape
    #         assert(clip > 0)
    #         jittered_data = torch.clamp(sigma * torch.randn(B, N, C, device=pc.device), -1 * clip, clip)
    #         jittered_data[:, :, 3:] = 0
    #         jittered_data += pc
    #         return jittered_data
    #     else:
    #         return pc
# -----------------------------------------------------------------------------------------

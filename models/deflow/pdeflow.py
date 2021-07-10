
import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from torch.nn import functional as F
from typing import List

from modules.utils.distribution import GaussianDistribution
from modules.linear import SoftClampling
from modules.augment import AugmentLayer, AugmentStep
from modules.normalize import ActNorm
from modules.permutate import Permutation
from modules.coupling import AffineCouplingLayer

from models.deflow.layer import KnnConvUnit, LinearUnit, AugmentShallow, get_knn_idx


# -----------------------------------------------------------------------------------------
class FlowAssembly(nn.Module):
    
    def __init__(self, idim, hdim, id):
        super(FlowAssembly, self).__init__()

        channel1 = idim - idim // 2
        channel2 = idim // 2

        self.actnorm = ActNorm(idim, dim=2)
        self.permutate1 = Permutation('reverse', idim, dim=2)
        self.permutate2 = Permutation('reverse', idim, dim=2)

        # if id < 3:
        #     self.coupling1 = AffineCouplingLayer('affine', KnnConvUnit, split_dim=2, clamp=SoftClampling(),
        #         params={ 'in_channel': channel1, 'hidden_channel': hdim, 'out_channel': channel2, })
        #     self.coupling2 = AffineCouplingLayer('affine', KnnConvUnit, split_dim=2, clamp=SoftClampling(),
        #         params={ 'in_channel': channel1, 'hidden_channel': hdim, 'out_channel': channel2, })
        # else:
        #     self.coupling1 = AffineCouplingLayer('affine', LinearUnit, split_dim=2, clamp=None,
        #         params={ 'in_channel': channel1, 'hidden_channel': hdim, 'out_channel': channel2, 'batch_norm': True })
        #     self.coupling2 = AffineCouplingLayer('affine', LinearUnit, split_dim=2, clamp=None,
        #         params={ 'in_channel': channel1, 'hidden_channel': hdim, 'out_channel': channel2, 'batch_norm': True })

        self.coupling1 = AffineCouplingLayer('affine', LinearUnit, split_dim=2, clamp=SoftClampling(),
            params={ 'in_channel': channel1, 'hidden_channel': hdim, 'out_channel': channel2, 'batch_norm': True })
        self.coupling2 = AffineCouplingLayer('affine', KnnConvUnit, split_dim=2, clamp=SoftClampling(),
            params={ 'in_channel': channel1, 'hidden_channel': hdim, 'out_channel': channel2, })

    def forward(self, x: Tensor, c: Tensor=None, knn_idx=None):
        x, _log_det2 = self.permutate1(x, c)
        x, _log_det0 = self.actnorm(x)
        x, _log_det1 = self.coupling1(x, c, knn_idx=knn_idx)
        x, _log_det4 = self.permutate2(x, c)
        x, _log_det3 = self.coupling2(x, c, knn_idx=knn_idx)

        if _log_det2 is not None:
            return x, _log_det0 + _log_det1 + _log_det2 + _log_det3 + _log_det4
        else:
            return x, _log_det0 + _log_det1 + _log_det3

    def inverse(self, z: Tensor, c: Tensor=None, knn_idx=None):
        z = self.coupling2.inverse(z, c, knn_idx=knn_idx)
        z = self.permutate2.inverse(z, c)
        z = self.coupling1.inverse(z, c, knn_idx=knn_idx)
        z = self.actnorm.inverse(z)
        z = self.permutate1.inverse(z, c)

        return z


# -----------------------------------------------------------------------------------------
class PointOutlierPooling(nn.Module):

    def __init__(self, pc_channel, aug_channel, hchannel, percent):
        super(PointOutlierPooling, self).__init__()

        in_channel = pc_channel + aug_channel
        self.outlier_percent = percent

        self.linears = nn.Sequential(
            nn.Linear(in_channel + pc_channel, hchannel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hchannel, hchannel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hchannel, 32),
            nn.ReLU(inplace=True))
        self.prob_linear = nn.Linear(32, 1)

        self.displacement_mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // 2, in_channel // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // 4, 3))

    def forward(self, xyz: Tensor, f: Tensor):
        """
        xyz: [B, N, C1]
        f  : [B, N, C2]
        return: [B, N]
        """
        B, N, _ = xyz.shape
        idxb = torch.arange(B).view(-1, 1)

        x = torch.cat([f, xyz], dim=-1)
        x = self.linears(x)  # [B, N, 32]

        x = self.prob_linear(x) / torch.norm(self.prob_linear.weight, p='fro')

        # f -> X, outlier_probs -> s, prob_idx -> i
        outlier_probs = torch.squeeze(x, dim=-1)  # [B, N]
        prob_idx = torch.argsort(outlier_probs, dim=-1, descending=True)  # [B, N]

        num_outlier = int(N * self.outlier_percent)
        clean_idx = prob_idx[:, num_outlier:]  # [B, N - _N]
        y = outlier_probs[idxb, clean_idx]     # [B, N - _N]
        y = torch.sigmoid(y)
        px = f[idxb, clean_idx]
        px = px * y.unsqueeze(-1).expand_as(px)

        displacement = self.displacement_mlp(px)
        est_xyz = xyz[idxb, clean_idx] + displacement # [B, N - _N, 3]
        return prob_idx, est_xyz


# -----------------------------------------------------------------------------------------
class ExDenoiseFlow(nn.Module):

    def __init__(self, pc_channel=3):
        super(ExDenoiseFlow, self).__init__()

        self.nflow_module = 12
        self.in_channel = pc_channel
        self.aug_channel = 21
        self.noise_channel = 3
        self.outlier_channel = 6

        self.dist = GaussianDistribution()

        # Augment Component
        shallow = AugmentShallow(pc_channel, hidden_channel=32, out_channel=64, num_convs=2)
        augment_steps = nn.ModuleList([
            AugmentStep(self.aug_channel, hidden_channel=64, reverse=False),
            AugmentStep(self.aug_channel, hidden_channel=64, reverse=True),
            AugmentStep(self.aug_channel, hidden_channel=64, reverse=False),
        ])
        self.argument = AugmentLayer(self.dist, self.aug_channel, shallow, augment_steps)

        self.outlier_percent = 0.25
        self.outlier_pooling = PointOutlierPooling(pc_channel, self.aug_channel, hchannel=64, percent=self.outlier_percent)

        # Flow Component
        self.pre_ks = [8, 16, 24]
        flow_assemblies = []

        for i in range(self.nflow_module):
            flow = FlowAssembly(self.in_channel + self.aug_channel, hdim=64, id=i)
            flow_assemblies.append(flow)
        self.flow_assemblies = nn.ModuleList(flow_assemblies)

    def f(self, x: Tensor, xyz: Tensor):
        log_det_J = torch.zeros((x.shape[0],), device=x.device)
        idxes = []

        for i in range(self.nflow_module):
            if i < len(self.pre_ks):
                knn_idx = get_knn_idx(k=self.pre_ks[i], f=xyz)
            else:
                knn_idx = get_knn_idx(k=16, f=x, q=None, offset=None)
                # knn_idx = None
            idxes.append(knn_idx)

            x, _log_det_J = self.flow_assemblies[i](x, c=None, knn_idx=knn_idx)
            if _log_det_J is not None:
                log_det_J += _log_det_J

            if i == 2:
                prob_idx, est_xyz = self.outlier_pooling(xyz, x)

        return x, est_xyz, log_det_J, idxes, prob_idx

    def g(self, z: Tensor, idxes: List[Tensor]):
        for i in reversed(range(self.nflow_module)):
            idx = idxes[i]
            z = self.flow_assemblies[i].inverse(z, c=None, knn_idx=idx)
        return z

    def log_prob(self, xyz: Tensor):

        y, aug_ldj = self.argument(xyz)
        x = torch.cat([xyz, y], dim=-1)  # [B, N, 3 + C]
        z, est_xyz, flow_ldj, idxes, prob_idx = self.f(x, xyz)
        logpz = self.dist.log_prob(z)

        logp = logpz + flow_ldj - aug_ldj
        logp = self.nll_loss(logp)
        return z, est_xyz, logp, idxes, prob_idx

    def sample(self, z: Tensor, idxes: List[Tensor]):
        full_x = self.g(z, idxes)
        clean_x = full_x[..., :self.in_channel]  # [B, N, 3]
        return clean_x

    def forward(self, xyz: Tensor):
        z, est_xyz, ldj, idxes, prob_idx = self.log_prob(xyz)

        noise_z = self.mask_outlier(z, prob_idx)
        clean_z = self.mask_noise(noise_z)
        clean_x = self.sample(clean_z, idxes)
        return clean_x, ldj, est_xyz

    def nll_loss(self, sldj):
        nll = -torch.mean(sldj)
        return nll

    def mask_noise(self, z: Tensor):
        # Fix channel mask
        z[:, :, -self.noise_channel:] = 0
        return z

    def mask_outlier(self, z: Tensor, prob_idx: Tensor):
        """
        z: [B, N, C]
        probs: [B, N]
        xyz: [B, N, 3]
        """
        B, N, _ = z.shape
        idxb = torch.arange(B).view(-1, 1)
        num_outlier = int(N * self.outlier_percent)

        outlier_idx = prob_idx[:, :num_outlier]  # [B, _N]

        # Fix channel mask
        z[idxb, outlier_idx, -self.outlier_channel:] = 0

        return z

    def init_as_trained_state(self):
        """Set the network to initialized state, needed for evaluation(significant performance impact)"""
        for i in range(self.nflow_module):
            self.flow_assemblies[i].actnorm.is_inited = True
# -----------------------------------------------------------------------------------------

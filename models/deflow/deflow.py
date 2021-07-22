
import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
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
        self.permutate1 = Permutation('inv1x1', idim, dim=2)
        self.permutate2 = Permutation('reverse', idim, dim=2)

        if id < 3 or id % 3 == 0:
            self.coupling1 = AffineCouplingLayer('affine', KnnConvUnit, split_dim=2, clamp=SoftClampling(),
                params={ 'in_channel': channel1, 'hidden_channel': hdim, 'out_channel': channel2, })
            self.coupling2 = AffineCouplingLayer('affine', KnnConvUnit, split_dim=2, clamp=SoftClampling(),
                params={ 'in_channel': channel1, 'hidden_channel': hdim, 'out_channel': channel2, })
        else:
            self.coupling1 = AffineCouplingLayer('affine', LinearUnit, split_dim=2, clamp=SoftClampling(),
                params={ 'in_channel': channel1, 'hidden_channel': hdim, 'out_channel': channel2, 'batch_norm': True })
            self.coupling2 = AffineCouplingLayer('affine', LinearUnit, split_dim=2, clamp=SoftClampling(),
                params={ 'in_channel': channel1, 'hidden_channel': hdim, 'out_channel': channel2, 'batch_norm': True })

    def forward(self, x: Tensor, c: Tensor=None, knn_idx=None):
        x, _log_det2 = self.permutate1(x, c)
        x, _log_det0 = self.actnorm(x)
        x, _log_det1 = self.coupling1(x, c, knn_idx=knn_idx)
        x, _log_det4 = self.permutate2(x, c)
        x, _log_det3 = self.coupling2(x, c, knn_idx=knn_idx)

        return x, _log_det0 + _log_det1 + _log_det2 + _log_det3

    def inverse(self, z: Tensor, c: Tensor=None, knn_idx=None):
        z = self.coupling2.inverse(z, c, knn_idx=knn_idx)
        z = self.permutate2.inverse(z, c)
        z = self.coupling1.inverse(z, c, knn_idx=knn_idx)
        z = self.actnorm.inverse(z)
        z = self.permutate1.inverse(z, c)

        return z


# -----------------------------------------------------------------------------------------
class DenoiseFlow(nn.Module):

    def __init__(self, pc_channel=3):
        super(DenoiseFlow, self).__init__()

        self.nflow_module = 12
        self.in_channel = pc_channel
        self.aug_channel = 20  # 20
        self.cut_channel = 3   # 3

        self.dist = GaussianDistribution()

        # Augment Component
        shallow = AugmentShallow(pc_channel, hidden_channel=32, out_channel=64, num_convs=2)
        augment_steps = nn.ModuleList([
            AugmentStep(self.aug_channel, hidden_channel=64, reverse=False),
            AugmentStep(self.aug_channel, hidden_channel=64, reverse=True),
            AugmentStep(self.aug_channel, hidden_channel=64, reverse=False),
        ])
        self.argument = AugmentLayer(self.dist, self.aug_channel, shallow, augment_steps)

        # Flow Component
        self.pre_ks = [8, 16, 24]
        flow_assemblies = []

        for i in range(self.nflow_module):
            flow = FlowAssembly(self.in_channel + self.aug_channel, hdim=64, id=i)
            flow_assemblies.append(flow)
        self.flow_assemblies = nn.ModuleList(flow_assemblies)

        # Channel Mask ----------------------------------
        # Fix channel mask
        self.channel_mask = nn.Parameter(torch.ones((1, 1, self.in_channel + self.aug_channel)), requires_grad=False)
        self.channel_mask[:, -self.cut_channel:] = 0.0

        # Random initialization (Works, but slow to convergence)
        # w_init = np.random.randn(self.in_channel + self.aug_channel, self.in_channel + self.aug_channel)
        # w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        # self.channel_mask = nn.Parameter(torch.from_numpy(w_init), requires_grad=True)

        # Identity initialization
        # w_init = torch.eye(self.in_channel + self.aug_channel)
        # w_init[-self.cut_channel:] = 0.0
        # w_init = w_init + torch.randn_like(w_init) * 0.05
        # self.channel_mask = nn.Parameter(w_init, requires_grad=True)

        # Learnable binary mask
        # theta = torch.rand((1, 1, self.in_channel + self.aug_channel))
        # self.theta = nn.Parameter(theta, requires_grad=True)
        # -----------------------------------------------

    def f(self, x: Tensor, xyz: Tensor):
        log_det_J = torch.zeros((x.shape[0],), device=x.device)
        idxes = []

        for i in range(self.nflow_module):
            if i < len(self.pre_ks):
                knn_idx = get_knn_idx(k=self.pre_ks[i], f=xyz, q=None)
            elif i % 3 == 0:
                knn_idx = get_knn_idx(k=16, f=x, q=None)
            else:
                knn_idx = None
            idxes.append(knn_idx)

            x, _log_det_J = self.flow_assemblies[i](x, c=None, knn_idx=knn_idx)
            if _log_det_J is not None:
                log_det_J += _log_det_J

        return x, log_det_J, idxes

    def g(self, z: Tensor, idxes: List[Tensor]):
        for i in reversed(range(self.nflow_module)):
            idx = idxes[i]
            z = self.flow_assemblies[i].inverse(z, c=None, knn_idx=idx)
        return z

    def log_prob(self, xyz: Tensor):

        y, aug_ldj = self.argument(xyz)
        x = torch.cat([xyz, y], dim=-1)  # [B, N, 3 + C]
        z, flow_ldj, idxes = self.f(x, xyz)
        logpz = self.dist.log_prob(z)

        logp = logpz + flow_ldj - aug_ldj
        logp = self.nll_loss(x.shape, logp)
        return z, logp, idxes

    def sample(self, z: Tensor, idxes: List[Tensor]):
        full_x = self.g(z, idxes)
        clean_x = full_x[..., :self.in_channel]  # [B, N, 3]
        return clean_x

    def forward(self, x: Tensor):
        z, ldj, idxes = self.log_prob(x)
        mask = None

        # Fix channel mask
        z[:, :, -self.cut_channel:] = 0
        clean_z = z
        # Identity initialization
        # clean_z = torch.einsum('ij,bnj->bni', self.channel_mask, z)
        # Random initialization
        # clean_z = z * self.channel_mask.expand_as(z)
        # Learnable binary mask
        # mask = torch.max(torch.zeros_like(self.theta), 1.0 - (-self.theta).exp())
        # clean_z = z * mask

        # TODO: transform clean point to latent code and compare mask loss

        clean_x = self.sample(clean_z, idxes)
        return clean_x, ldj, mask

    def nll_loss(self, pts_shape, sldj):
        #ll = sldj - np.log(self.k) * torch.prod(pts_shape[1:])
        # ll = torch.nan_to_num(sldj, nan=1e3)
        ll = sldj

        nll = -torch.mean(ll)

        return nll

    def init_as_trained_state(self):
        """Set the network to initialized state, needed for evaluation(significant performance impact)"""
        for i in range(self.nflow_module):
            self.flow_assemblies[i].actnorm.is_inited = True
# -----------------------------------------------------------------------------------------

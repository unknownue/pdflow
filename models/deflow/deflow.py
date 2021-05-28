
import torch

from torch import nn
from torch.tensor import Tensor

from modules.augment import AugmentLayer, AugmentStep
from modules.linear import LinearNN1D
from modules.coupling import AffineCouplingLayer
from modules.normalize import ActNorm
from modules.permutate import Permutation
from modules.sequential import FlowSequential
from modules.utils.distribution import GaussianDistribution


# -----------------------------------------------------------------------------------------
class FlowStep(nn.Module):

    def __init__(self, idim, hdim):
        super(FlowStep, self).__init__()

        self.actnorm   = ActNorm(idim)
        self.permutate = Permutation('inv1x1', idim, dim=1)   # inv1x1 > random

        self.coupling1 = AffineCouplingLayer('affine', LinearNN1D, split_dim=1, clamp=None,
            params={ 'in_channels' : idim // 2, 'hidden_channels': hdim, 'out_channels': idim - idim // 2 })
        self.coupling2 = AffineCouplingLayer('affine', LinearNN1D, split_dim=1, clamp=None,
            params={ 'in_channels' : idim // 2, 'hidden_channels': hdim, 'out_channels': idim - idim // 2 })

    def forward(self, x: Tensor, c: Tensor=None):
        x, _log_det0 = self.actnorm(x)
        x, _log_det1 = self.permutate(x, c)
        x, _log_det2 = self.coupling1(x, c)
        x, _log_det3 = self.coupling2(x, c)

        return x, _log_det0 + _log_det1 + _log_det2 + _log_det3

    def inverse(self, z: Tensor, c: Tensor=None):
        z = self.coupling2.inverse(z, c)
        z = self.coupling1.inverse(z, c)
        z = self.permutate.inverse(z, c)
        z = self.actnorm.inverse(z)
        return z


# -----------------------------------------------------------------------------------------
class DenoiseFlow(nn.Module):

    def __init__(self):
        super(DenoiseFlow, self).__init__()

        self.nflow_step = 10
        self.in_channel  = 2
        self.aug_channel = 6
        self.cut_channel = 2

        self.dist = GaussianDistribution()

        shallow = LinearNN1D(in_channels=self.in_channel, hidden_channels=64, out_channels=64)
        augment_steps = nn.ModuleList([
            AugmentStep(self.aug_channel, 64),
            AugmentStep(self.aug_channel, 64),
            AugmentStep(self.aug_channel, 64),
            AugmentStep(self.aug_channel, 64),
        ])
        self.argument = AugmentLayer(self.dist, self.aug_channel, shallow, augment_steps)

        steps = []
        for _ in range(self.nflow_step):
            step = FlowStep(idim=(self.in_channel + self.aug_channel), hdim=64)
            steps.append(step)

        self.flow_layers = FlowSequential(steps)

        self.mask = torch.ones((1, self.in_channel + self.aug_channel, 1), requires_grad=False)
        self.mask[:, -self.cut_channel:] = 0

    def f(self, x: Tensor):
        return self.flow_layers(x, c=None)

    def g(self, z: Tensor):
        return self.flow_layers.inverse(z, c=None)

    def log_prob(self, x: Tensor):
        y, aug_ldj = self.argument(x)
        x = torch.cat([x, y], dim=1)
        z, ldj = self.f(x)
        logpz = self.dist.log_prob(z)

        logp = logpz + ldj - aug_ldj
        logp = self.nll_loss(x.shape, logp)
        return z, logp

    def forward(self, x: Tensor):
        z, ldj = self.log_prob(x)

        # mask = self.mask.to(z.device)
        # clean_z = z * mask.expand_as(z)
        z[:, -self.cut_channel:] = 0
        clean_z = z
        clean_x = self.sample(clean_z)
        return clean_x, ldj

    def sample(self, z: Tensor):
        full_x = self.g(z)
        return full_x[:, :self.in_channel, ...]

    def nll_loss(self, pts_shape, sldj):
        #ll = sldj - np.log(self.k) * torch.prod(pts_shape[1:])
        # ll = torch.nan_to_num(sldj, nan=1e3)
        ll = sldj

        nll = -torch.mean(ll)

        return nll
# -----------------------------------------------------------------------------------------


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
class AugmentFlow(nn.Module):

    def __init__(self):
        super(AugmentFlow, self).__init__()

        self.nflow_step = 10
        self.in_channel  = 2
        self.aug_channel = 6

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

    def f(self, x: Tensor):
        return self.flow_layers(x, c=None)

    def g(self, z: Tensor):
        return self.flow_layers.inverse(z, c=None)

    def log_prob(self, x: Tensor):
        y, aug_ldj = self.argument(x)
        x = torch.cat([x, y], dim=1)
        z, ldj = self.f(x)
        logpx = self.dist.log_prob(z)

        logp = logpx + ldj - aug_ldj
        return self.nll_loss(x.shape, logp)

    def forward(self, x: Tensor):
        return self.log_prob(x)

    def sample(self, shape, device):
        B, _, N = shape
        z = torch.randn((B, self.in_channel + self.aug_channel, N), device=device)
        full_x = self.g(z)
        return full_x[:, :self.in_channel, ...]

    def nll_loss(self, pts_shape, sldj):
        #ll = sldj - np.log(self.k) * torch.prod(pts_shape[1:])
        # ll = torch.nan_to_num(sldj, nan=1e3)
        ll = sldj

        nll = torch.mean(-ll)

        return nll
# -----------------------------------------------------------------------------------------

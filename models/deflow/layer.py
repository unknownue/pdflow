
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F




# -----------------------------------------------------------------------------------------
def knn_group(x: Tensor, i: Tensor):
    """
    x: [B, N, C]
    i: [B, M, k]
    return: [B, M, k, C]
    """
    (B, N, C), (_, M, k) = x.shape, i.shape

    x = x.unsqueeze(1).expand(B, M, N, C)
    i = i.unsqueeze(3).expand(B, M, k, C)
    return torch.gather(x, dim=2, index=i)

# -----------------------------------------------------------------------------------------
def get_knn_idx(k: int, f: Tensor, q: Tensor=None, offset=None, return_features=True):
    """
    f: [B, N, C]
    q: [B, M, C]
    index of points in f: [B, M, k]
    """
    if offset is None:
        offset = 0
    if q is None:
        q = f

    (B, N, C), (_, M, _) = f.shape, q.shape

    _f = f.unsqueeze(1).expand(B, M, N, C)
    _q = q.unsqueeze(2).expand(B, M, N, C)

    # TODO: Fix high memory issue
    dist = torch.sum((_f - _q) ** 2, dim=3, keepdim=False)  # [B, M, N]
    knn_idx = torch.argsort(dist, dim=2)[..., offset:k+offset]  # [B, M, k]

    if return_features is True:
        knn_f = knn_group(f, knn_idx)
        return knn_f, knn_idx
    else:
        return knn_idx


# -----------------------------------------------------------------------------------------
class KnnConvUnit(nn.Module):

    def __init__(self, in_channel, hidden_channel, out_channel, k=None, bias=True):
        super(KnnConvUnit, self).__init__()

        self.k = k

        self.linear1 = nn.Linear(in_channel * 3, hidden_channel, bias=bias)
        self.linear2 = nn.Linear(hidden_channel, hidden_channel, bias=bias)
        self.linear3 = nn.Linear(hidden_channel, out_channel, bias=bias)

        nn.init.normal_(self.linear3.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.linear3.bias)

    def forward(self, f: Tensor, _c=None, knn_idx: Tensor=None):
        """
        f: [B, N, C]
        knn_idx: [B, N, k]
        return: [B, N, out]
        """

        if knn_idx is None:
            knn_feat, _ = get_knn_idx(self.k, f, q=None, offset=None, return_features=True)
        else:
            knn_feat = knn_group(f, knn_idx)  # [B, M, k, C]

        f_tiled = f.unsqueeze(2).expand_as(knn_feat)  # [B, N, k, C]
        x = torch.cat([f_tiled, knn_feat, knn_feat - f_tiled], dim=-1)  # [B, N, k, C * 3]

        x = F.relu(self.linear1(x))  # [B, N, k, h]
        x = F.relu(self.linear2(x))  # [B, N, k, h]
        x, _ = torch.max(x, dim=2, keepdim=False)  # [B, N, h]
        x = self.linear3(x)   # [B, N, out]

        return x


# -----------------------------------------------------------------------------------------
class AugmentShallow(nn.Module):
    
    def __init__(self, in_channel, hidden_channel, out_channel, num_convs=2):
        super(AugmentShallow, self).__init__()

        self.trans1 = nn.Linear(in_channel, hidden_channel)
        self.trans2 = nn.Linear(hidden_channel, out_channel)

        self.convs = nn.ModuleList()
        for _ in range(num_convs):
            conv = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=(1, 1))
            self.convs.append(conv)

    def forward(self, x: Tensor, knn_idx: Tensor=None):
        """
        x: [B, N, C]
        knn_idx: [B, N, k]
        """
        if knn_idx is None:
            knn_idx = get_knn_idx(12, x, return_features=False)

        x = knn_group(x, knn_idx)   # [B, N, k, C]
        x = self.trans1(x)   # [B, N, h]
        x = torch.transpose(x, 1, -1)  # [B, h, k, N]

        for conv_layer in self.convs:
            x = F.relu(conv_layer(x))  # [B, h, k, N]

        x = torch.mean(x, dim=2, keepdim=False)  # [B, h, N]
        x = torch.transpose(x, 1, 2)  # [B, N, h]
        x = self.trans2(x)

        return x  # [B, N, o]
# -----------------------------------------------------------------------------------------

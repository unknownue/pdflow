
import os
import torch

from torch import nn

from contextlib import redirect_stdout
with open(os.devnull, "w") as outer_space, redirect_stdout(outer_space):
    from metric.PyTorchCD.chamfer2D import dist_chamfer_2D



# -----------------------------------------------------------------------------------------
class ChamferDistance2D(nn.Module):

    def __init__(self):
        super(ChamferDistance2D, self).__init__()
        self.chamLoss = dist_chamfer_2D.chamfer_2DDist()

    def forward(self, points1, points2):
        dist1, dist2, _, _ = self.chamLoss(points1, points2)
        cost = torch.mean(dist1, dim=-1) + torch.mean(dist2, dim=-1)
        return torch.sum(cost)


# -----------------------------------------------------------------------------------------
class DenoiseMetrics(object):

    def __init__(self):
        super(DenoiseMetrics, self).__init__()
        self.cd_loss = ChamferDistance2D()

    def evaluate(self, pred, gt):
        result = {}
        result['CD'] = self.cd_loss(pred.transpose(1, 2), gt.transpose(1, 2))

        return result
# -----------------------------------------------------------------------------------------

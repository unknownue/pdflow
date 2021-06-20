
import os
import torch

from torch import nn, Tensor
from torch.nn import functional as F

from metric.emd.emd_module import emdFunction
from modules.utils.fps import square_distance


# -----------------------------------------------------------------------------------------
class EarthMoverDistance(nn.Module):

    def __init__(self, eps=0.005, iters=50):
        super().__init__()
        self.eps = eps
        self.iters = iters

    def forward(self, preds, gts, **kwargs):
        loss, _ = emdFunction.apply(preds, gts, self.eps, self.iters)
        return torch.sum(loss)


# -----------------------------------------------------------------------------------------
class ChamferDistance(nn.Module):

    def __init__(self, dim):
        super(ChamferDistance, self).__init__()
        assert dim in [2, 3]

        if dim == 2:
            from contextlib import redirect_stdout
            with open(os.devnull, "w") as outer_space, redirect_stdout(outer_space):
                from metric.PyTorchCD.chamfer2D import dist_chamfer_2D
                self.chamLoss = dist_chamfer_2D.chamfer_2DDist()
        elif dim == 3:
            from contextlib import redirect_stdout
            with open(os.devnull, "w") as outer_space, redirect_stdout(outer_space):
                from metric.PyTorchCD.chamfer3D import dist_chamfer_3D
            self.chamLoss = dist_chamfer_3D.chamfer_3DDist()

    def forward(self, points1: Tensor, points2: Tensor):
        """
        points1: [B, N1, C]
        points2: [B, N2, C]
        confi  : [B, N1],
        """

        # points1 = points1.transpose(1, 2)
        # points2 = points2.transpose(1, 2)

        # CD
        dist1, dist2, _, _ = self.chamLoss(points1, points2)  # square distance, [B, N1], [B, N2]
        cost = torch.mean(dist1, dim=-1) + torch.mean(dist2, dim=-1)
        loss_cd = torch.sum(cost)

        # # Confidence
        # square_dis = square_distance(points1, points2)  # [B, N, N]
        # max_distance = torch.max(torch.max(square_dis, dim=-1)[0], dim=-1, keepdim=True)[0]  # [B, 1]

        # # tmp1 = torch.exp(dist1)
        # # tmp2 = torch.exp(max_distance)
        # # gt_confidence = 1.0 - torch.exp(dist1) / torch.exp(max_distance)  # [B, N1]
        # # gt_confidence = 1.0 - dist1
        # gt_confidence = 1.0 - dist1 / max_distance  # [B, N1]
        # loss_confi = F.mse_loss(confi, gt_confidence)

        return loss_cd

# -----------------------------------------------------------------------------------------
class NoiseConfidence(nn.Module):

    def forward(self, pred_confi: Tensor, label_confi: Tensor):
        prob = torch.sigmoid(pred_confi)
        return F.binary_cross_entropy(prob, label_confi)


# -----------------------------------------------------------------------------------------
class DenoiseMetrics(object):

    def __init__(self, dim=3):
        super(DenoiseMetrics, self).__init__()
        self.cd_loss = ChamferDistance(dim)
        self.confid_loss = NoiseConfidence()

    def evaluate(self, pred_pt, gt_pt, pred_probs, gt_probs):
        result = {}
        result['CD']    = self.cd_loss(pred_pt, gt_pt)
        result['Confi'] = self.confid_loss(pred_probs, gt_probs)

        return result
# -----------------------------------------------------------------------------------------

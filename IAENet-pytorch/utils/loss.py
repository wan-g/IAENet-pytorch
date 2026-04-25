import torch
import torch.nn.functional as F
import torch as t
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Optional
from torch.nn.modules.loss import _Loss


class LCRLoss(nn.Module):
    def __init__(self, co_matrix: torch.Tensor, label_priors, weight_strategy='none', lambda_co=0.02, eps=1e-8, reduction='mean'):
        """
                label_priors: shape (num_labels,)  -> 每个标签为正样本的概率 P(y=1)
                weight_strategy:
                    'inverse'      -> 正样本权重 1/p，负样本权重 1/(1-p)
                    'log_inverse'  -> 正样本权重 log(1/p)，负样本权重 log(1/(1-p))
                    'sqrt_inverse' -> 正样本权重 1/sqrt(p)，负样本权重 1/sqrt(1-p)
                    'cubic_inverse' -> 正样本权重 1/cubic(p)，负样本权重 1/cubic(1-p)
                    'none'         -> 不使用任何权重
                """
        super().__init__()
        self.C = co_matrix  # shape: (C, C)
        self.lambda_co = lambda_co
        self.eps = eps
        self.reduction = reduction

        self.weight_strategy = weight_strategy.lower()

        if self.weight_strategy == 'inverse':
            self.pos_weight = 1.0 / label_priors   # e.g. 1 / pos_ratio
            self.neg_weight = 1.0 / (1.0 - label_priors)  # e.g. 1 / neg_ratio
        elif self.weight_strategy == 'sqrt_inverse':
            self.pos_weight = 1.0 / torch.sqrt(label_priors)  # torch.sqrt
            self.neg_weight = 1.0 / torch.sqrt(1.0 - label_priors)
        elif self.weight_strategy == 'cubic_inverse':
            self.pos_weight = 1.0 / torch.pow(label_priors, 1 / 3)
            self.neg_weight = 1.0 / torch.pow(1.0 - label_priors, 1 / 3)
        elif self.weight_strategy == 'log_inverse':
            self.pos_weight = 1.0 / torch.log(label_priors)  # torch.log
            self.neg_weight = 1.0 / torch.log(1.0 - label_priors)
        elif self.weight_strategy == 'none':
            self.pos_weight = torch.ones_like(label_priors)  # torch.log
            self.neg_weight = torch.ones_like(label_priors)
        else:
            raise ValueError(f"Unsupported weight strategy: {self.weight_strategy}")


    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)  # (N, C)
        ce = - (self.pos_weight * targets * torch.log(probs + self.eps) +
                self.neg_weight * (1 - targets) * torch.log(1 - probs + self.eps))  # (N, C)

        # Co-occurrence regularization
        B, C = logits.shape
        co_loss = 0.0
        for i in range(C):
            for j in range(C):
                if i != j:
                    # 强调共现关系强的标签预测要更一致（L2距离小）
                    co_loss += self.C[i, j] * ((logits[:, i] - logits[:, j]) ** 2).mean()

        total_loss = ce + self.lambda_co * co_loss / (C * C)

        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss

class PriorWeightedBCELoss(nn.Module):
    def __init__(self, label_priors, weight_strategy='log_inverse', reduction='mean', eps=1e-6):
        """
        label_priors: shape (num_labels,)  -> 每个标签为正样本的概率 P(y=1)
        weight_strategy:
            'inverse'      -> 正样本权重 1/p，负样本权重 1/(1-p)
            'log_inverse'  -> 正样本权重 log(1/p)，负样本权重 log(1/(1-p))
            'sqrt_inverse' -> 正样本权重 1/sqrt(p)，负样本权重 1/sqrt(1-p)
            'none'         -> 不使用任何权重
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.weight_strategy = weight_strategy.lower()

        if self.weight_strategy == 'none':
            self.register_buffer('pos_weight', torch.ones_like(label_priors))
            self.register_buffer('neg_weight', torch.ones_like(label_priors))
        elif self.weight_strategy == 'inverse':
            self.register_buffer('pos_weight', 1.0 / (label_priors + eps))
            self.register_buffer('neg_weight', 1.0 / (1.0 - label_priors + eps))
        elif self.weight_strategy == 'log_inverse':
            self.register_buffer('pos_weight', torch.log(1.0 / (label_priors + eps)))
            self.register_buffer('neg_weight', torch.log(1.0 / (1.0 - label_priors + eps)))
        elif self.weight_strategy == 'sqrt_inverse':
            self.register_buffer('pos_weight', 0.5 / torch.sqrt(label_priors + eps))
            self.register_buffer('neg_weight', 1.0 / torch.sqrt(1.0 - label_priors + eps))
        elif self.weight_strategy == 'cubic_inverse':
            self.register_buffer('pos_weight', 1.0 / torch.pow(label_priors + eps, 1 / 3))
            self.register_buffer('neg_weight', 1.0 / torch.pow(1.0 - label_priors + eps, 1 / 3))
        else:
            raise ValueError(f"Unsupported weight strategy: {self.weight_strategy}")


    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        loss = - (self.pos_weight * targets * torch.log(probs + self.eps) +
                  self.neg_weight * (1 - targets) * torch.log(1 - probs + self.eps))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FocalBCELoss(nn.Module):

    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, gt):
        bce_loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class FocalBCELossv1(nn.Module):
    """
       Implementation of Focal loss.
       Refers to Focal loss for dense object detection (ICCV 2017)
       <https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html>
       """
    def __init__(self, alpha=1.00, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, gt):
        pred_sigmoid = torch.sigmoid(pred)
        pt = pred_sigmoid * gt + (1 - pred_sigmoid) * (1 - gt)  # pt for focal
        alpha_t = self.alpha * gt + (1 - self.alpha) * (1 - gt)  # alpha weighting

        bce_loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Implementation of ASL loss.
    Refers to Asymmetric loss for multi-label classification (ICCV 2021)
    <https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html>
    """
    def __init__(self, gamma_pos=0.5, gamma_neg=3.0, eps=1e-8, reduction='mean'):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C] logits
        # targets: [B, C] 0 or 1

        inputs = torch.sigmoid(inputs)

        # Clamp to prevent log(0)
        inputs = torch.clamp(inputs, min=self.eps, max=1.0 - self.eps)

        pos_weight = torch.pow(1 - inputs, self.gamma_pos)
        neg_weight = torch.pow(inputs, self.gamma_neg)

        loss = - targets * pos_weight * torch.log(inputs) - \
               (1 - targets) * neg_weight * torch.log(1 - inputs)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class PolyLoss(torch.nn.Module):
    """
    Implementation of poly loss.
    Refers to PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions (ICLR 2022)
    <https://arxiv.org/abs/2204.12511>
    """

    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, target):
        bce = self.criterion(output, target)
        prob = torch.sigmoid(output)
        pt = prob * target + (1 - prob) * (1 - target)  # (B, C), per-class correctness
        poly_loss = bce + self.epsilon * (1 - pt)

        return poly_loss.mean()

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result

class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))

class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)

class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)

class BCEWithLogitsLossLabelSmoothing(_Loss):
    """
        Args:
            smoothing (float): 标签平滑因子，通常在0.0-0.2之间
            reduction (str): 指定如何聚合损失值，'mean'|'sum'|'none'
    """
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[Tensor] = None, smoothing=0.1) -> None:
        super(BCEWithLogitsLossLabelSmoothing, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.weight: Optional[Tensor]
        self.pos_weight: Optional[Tensor]
        self.smoothing = smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # label smoothing
        target = target * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch import nn
from torch.nn import functional as F
EPS = 1e-2
esp = 1e-8

class RankLoss(nn.Module):
    """Monotonicity regularization loss, will be zero when rankings of pred and target are the same.

    Reference:
        - https://github.com/lidq92/LinearityIQA/blob/master/IQAloss.py

    """

    def __init__(self, detach=False, loss_weight=1.0):
        super(RankLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        if pred.size(0) > 1:  #
            ranking_loss = F.relu((pred - pred.t()) * torch.sign((target.t() - target)))
            scale = 1 + torch.max(ranking_loss.detach())
            loss = ranking_loss.mean() / scale
        else:
            loss = F.l1_loss(pred, target.detach())  # 0 for batch with single sample.
        return self.loss_weight * loss
class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()
# def plcc_loss(pred, target):
#     """
#     Args:
#         pred (Tensor): of shape (N, 1). Predicted tensor.
#         target (Tensor): of shape (N, 1). Ground truth tensor.
#     """
#     batch_size = pred.shape[0]
#     if batch_size > 1:
#         vx = pred - pred.mean()
#         vy = target - target.mean()
#         loss = F.normalize(vx, p=2, dim=0) * F.normalize(vy, p=2, dim=0)
#         loss = (1 - loss.sum()) / 2  # normalize to [0, 1]
#     else:
#         loss = F.l1_loss(pred, target)
#     return loss.mean()
def loss_m3(y_pred, y):
    """prediction monotonicity related loss"""
    assert y_pred.size(0) > 1  #
    # print(y_pred.size())
    # y_pred = y_pred.unsqueeze(1)
    # print(y_pred.size())
    # y = y.unsqueeze(1)
    preds = y_pred-y_pred.t()
    gts = y - y.t()

    #signed = torch.sign(gts)

    triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
    preds = preds[triu_indices[0], triu_indices[1]]
    gts = gts[triu_indices[0], triu_indices[1]]
    g = 0.5 * (torch.sign(gts) + 1)

    constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
    p = 0.5 * (1 + torch.erf(preds / constant))

    g = g.view(-1, 1)
    p = p.view(-1, 1)

    loss = torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))

    return loss
def loss_m(y_pred, y):
    """prediction monotonicity related loss"""
    assert y_pred.size(0) > 1  #
    preds = y_pred-(y_pred + 10).t()
    gts = y.t() - y
    triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
    preds = preds[triu_indices[0], triu_indices[1]]
    gts = gts[triu_indices[0], triu_indices[1]]
    return torch.sum(F.relu(preds * torch.sign(gts))) / preds.size(0)
    #return torch.sum(F.relu((y_pred-(y_pred + 10).t()) * torch.sign((y.t()-y)))) / y_pred.size(0) / (y_pred.size(0)-1)

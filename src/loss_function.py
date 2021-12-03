import torch
from torch import nn, Tensor
from torch.nn.functional import logsigmoid


class LossFunction(nn.Module):
    def __init__(self, loss_type="TOP1"):
        super(LossFunction, self).__init__()

        if loss_type == "CrossEntropy":
            self._loss_fn = CrossEntropyLoss()
        elif loss_type == "TOP1":
            self._loss_fn = TOP1Loss()
        elif loss_type == "BPR":
            self._loss_fn = BPRLoss()
        elif loss_type == "TOP1-max":
            self._loss_fn = TOP1Max()
        elif loss_type == "BPR-max":
            self._loss_fn = BPRMax()
        else:
            raise NotImplementedError

    def forward(self, logit):
        return self._loss_fn(logit)


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    @staticmethod
    def forward(logit: Tensor) -> Tensor:
        loss = torch.mean(-torch.log(torch.diag(logit) + 1e-24))
        return loss


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    @staticmethod
    def forward(logit: Tensor) -> Tensor:
        diff = logit.diag().view(-1, 1).expand_as(logit).T - logit
        loss = -torch.mean(logsigmoid(diff))
        return loss


class BPRMax(nn.Module):
    def __init__(self):
        super(BPRMax, self).__init__()

    @staticmethod
    def forward(logit: Tensor) -> Tensor:
        logit_softmax = torch.softmax(logit, dim=1)
        diff = logit.diag().view(-1, 1).expand_as(logit).T - logit
        loss = -torch.log(torch.mean(logit_softmax * torch.sigmoid(diff)))
        return loss


class TOP1Loss(nn.Module):
    def __init__(self):
        super(TOP1Loss, self).__init__()

    @staticmethod
    def forward(logit: Tensor) -> Tensor:
        batch_size = logit.size(0)
        bpr = torch.mean(torch.sigmoid(logit - logit.diag().expand_as(logit).T), axis=1)
        l2 = torch.mean(torch.sigmoid(logit ** 2), axis=1)
        loss = torch.mean(bpr + l2 -torch.sigmoid(logit.diag() ** 2) / batch_size)
        return loss


class TOP1Max(nn.Module):
    def __init__(self):
        super(TOP1Max, self).__init__()

    @staticmethod
    def forward(logit: Tensor) -> Tensor:
        logit_softmax = torch.softmax(logit, dim=1)
        diff = logit - logit.diag().view(-1, 1).expand_as(logit).T
        loss = torch.mean(logit_softmax * (torch.sigmoid(diff) + torch.sigmoid(logit ** 2)))
        return loss

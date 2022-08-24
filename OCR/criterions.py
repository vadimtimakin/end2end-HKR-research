import torch
from torch import nn, Tensor
import torch.nn.functional as F


class FocalCTCLoss(nn.Module):
    blank: int
    zero_infinity: bool
    reduction: str

    def __init__(self, blank: int = 0, zero_infinity: bool = False, alpha: float = 1, gamma: float = 0.75):
        super(FocalCTCLoss, self).__init__()
        self.blank = blank
        self.zero_infinity = zero_infinity
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor) -> Tensor:
        # ctc_loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len, time_major=True)
        #     p= tf.exp(-ctc_loss)
        #     focal_ctc_loss= tf.multiply(tf.multiply(alpha,tf.pow((1-p),gamma)),ctc_loss) #((alpha)*((1-p)**gamma)*(ctc_loss))
        #     loss = tf.reduce_mean(focal_ctc_loss)
        ctc_loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, self.blank, 'none', self.zero_infinity) / target_lengths.to(log_probs.device)
        p = torch.exp(-ctc_loss)
        focal_ctc_loss = ctc_loss * (self.alpha * torch.pow(1 - p, self.gamma))
        return torch.mean(focal_ctc_loss, dim=0)


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device = None,
    dtype: torch.dtype = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1, gamma: float = 0.75, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input_, target):
        input_soft: torch.Tensor = F.softmax(input_, dim=1)
        log_input_soft: torch.Tensor = F.log_softmax(input_, dim=1)

        # create the labels one hot tensor
        target_one_hot: torch.Tensor = one_hot(target, num_classes=input_.shape[1], device=input_.device, dtype=input_.dtype)

        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1.0, self.gamma)

        focal = -self.alpha * weight * log_input_soft
        loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))
        loss_tmp = loss_tmp[target != self.ignore_index]
        return torch.mean(loss_tmp)

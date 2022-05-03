import torch.nn as nn
import torch.nn.functional as F
from trash.spike_blur import SpikeSmoothing
import torch


class FocalLossWithLogitsLoss(nn.Module):
    def __init__(self, gamma=1.0, alpha=0.25, pos_weight=None, reduction: str = "mean"):
        super(FocalLossWithLogitsLoss, self).__init__()
        self.register_buffer('gamma', torch.tensor(gamma))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('pos_weight', pos_weight)
        self.reduction = reduction

    def forward(self, input, target):
        p = torch.sigmoid(input)
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none', pos_weight=self.pos_weight)
        # pt = torch.exp(-ce_loss ) # prevents nans when probability 0
        pt = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


def bcel_mse_dvt_loss(loss_weights, window_size, sigma):
    def custom_loss(output, target):

        if output[0].device != target[0].device:
            min_len = min(len(target), len(output))
            for i in range(min_len):  # same processor for comperison
                target[i] = target[i].to(output[i].device)
        # binary_cross_entropy_loss = nn.functional.binary_cross_entropy_with_logits()
        mse_loss = nn.MSELoss()
        general_loss = None
        loss_bcel_item = 0
        loss_mse_item = 0
        loss_dvt_item = 0
        loss_blur_item = 0

        if loss_weights[0] > 0:
            loss_bcel = F.binary_cross_entropy_with_logits(output[0], target[0])  # removing channel dimention
            loss_bcel_item = loss_bcel.item()
            general_loss = loss_weights[0] * loss_bcel

        if loss_weights[1] > 0:
            loss_mse = mse_loss(output[1], target[1])
            loss_mse_item = loss_mse.item()
            general_loss = general_loss + loss_weights[1] * loss_mse if general_loss else loss_weights[1] * loss_mse

        if loss_weights[2] > 0:
            loss_dvt = mse_loss(output[2], target[2])
            loss_dvt_item = loss_dvt.item()
            general_loss = general_loss + loss_weights[2] * loss_dvt if general_loss else loss_weights[2] * loss_dvt

        # if len(loss_weights) > 3 and loss_weights[3] != 0 and False: #cannot be done on a single point
        #     pass
        #     g_blur = SpikeSmoothing(1, window_size, sigma, 1).to('cuda', torch.double)
        #     blur_output = (output[0] >= 0.5).double()
        #     loss_blur = loss_weights[3] * mse_loss(g_blur(blur_output.squeeze(3)), g_blur(target[0].squeeze(3)))
        #     general_loss = general_loss + loss_blur if general_loss else loss_blur
        return general_loss, loss_bcel_item, loss_mse_item, loss_dvt_item, loss_blur_item
        # return general_loss, 0, 0, loss_dvt

    return custom_loss

def weighted_mse_loss(loss_weights, window_size, sigma):
    def custom_loss(output, target):
        if output[0].device != target[0].device:
            min_len = min(len(target), len(output))
            for i in range(min_len):  # same processor for comperison
                target[i] = target[i].to(output[i].device)
        mse_loss = nn.MSELoss(reduction='none')
        weights = target[0]
        weights[1:]+=target[0][:-1]
        weights[:-1]+=target[0][1:]
        weights *=999
        weights+=1
        # weights[1:] = torch.pow(target[1][1:] - target[:-1], 2) + 1
        loss_mse = mse_loss(output[1], target[1])
        loss_mse = loss_mse * weights
        loss_mse = torch.mean(loss_mse)
        loss_mse_item = loss_mse.item()
        general_loss = loss_mse
        return general_loss, 0, loss_mse_item, 0, 0

    return custom_loss

def weighted_mse_loss_derivative(loss_weights, window_size, sigma):
    def custom_loss(output, target):
        if output[0].device != target[0].device:
            min_len = min(len(target), len(output))
            for i in range(min_len):  # same processor for comperison
                target[i] = target[i].to(output[i].device)
        mse_loss = nn.MSELoss(reduction='none')
        weights = torch.ones_like(target[1])
        # weights[1:] = torch.pow(target[1][1:] - target[:-1], 2) + 1
        weights[1:] = torch.abs(target[1][1:] - target[1][:-1]) + 1
        loss_mse = mse_loss(output[1], target[1])
        loss_mse = loss_mse * weights
        loss_mse = torch.mean(loss_mse)
        loss_mse_item = loss_mse.item()
        general_loss = loss_mse
        return general_loss, 0, loss_mse_item, 0, 0

    return custom_loss


def hinge_mse_dvt_loss(loss_weights, window_size, sigma):
    def custom_loss(output, target):

        if output[0].device != target[0].device:
            min_len = min(len(target), len(output))
            for i in range(min_len):  # same processor for comperison
                target[i] = target[i].to(output[i].device)
        hinge_loss = nn.HingeEmbeddingLoss()
        mse_loss = nn.MSELoss()
        general_loss = None
        loss_bcel_item = 0
        loss_mse_item = 0
        loss_dvt_item = 0
        loss_blur_item = 0

        if loss_weights[0] > 0:
            out_hinge = output[0] * 2 - 1
            tar_hinge = target[0] * 2 - 1
            loss_bcel = hinge_loss(out_hinge, tar_hinge)  # removing channel dimention
            loss_bcel_item = loss_bcel.item()
            general_loss = loss_weights[0] * loss_bcel

        if loss_weights[1] > 0:
            loss_mse = mse_loss(output[1], target[1])
            loss_mse_item = loss_mse.item()
            general_loss = general_loss + loss_weights[1] * loss_mse if general_loss else loss_weights[1] * loss_mse

        if loss_weights[2] > 0:
            loss_dvt = mse_loss(output[2], target[2])
            loss_dvt_item = loss_dvt.item()
            general_loss = general_loss + loss_weights[2] * loss_dvt if general_loss else loss_weights[2] * loss_dvt

        # if len(loss_weights) > 3 and loss_weights[3] != 0 and False: #cannot be done on a single point
        #     pass
        #     g_blur = SpikeSmoothing(1, window_size, sigma, 1).to('cuda', torch.double)
        #     blur_output = (output[0] >= 0.5).double()
        #     loss_blur = loss_weights[3] * mse_loss(g_blur(blur_output.squeeze(3)), g_blur(target[0].squeeze(3)))
        #     general_loss = general_loss + loss_blur if general_loss else loss_blur
        return general_loss, loss_bcel_item, loss_mse_item, loss_dvt_item, loss_blur_item
        # return general_loss, 0, 0, loss_dvt

    return custom_loss


def loss_zero_mse_on_spikes(loss_weights, window_size, sigma):
    def custom_loss(output, target):

        if output[0].device != target[0].device:
            min_len = min(len(target), len(output))
            for i in range(min_len):  # same processor for comperison
                target[i] = target[i].to(output[i].device)
        binary_cross_entropy_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()
        general_loss = None
        loss_bcel_item = 0
        loss_mse_item = 0
        loss_dvt_item = 0
        loss_blur_item = 0

        if loss_weights[0] > 0:
            loss_bcel = loss_weights[0] * binary_cross_entropy_loss(output[0],
                                                                    target[0])  # removing channel dimention
            loss_bcel_item = loss_bcel.item()
            general_loss = loss_bcel

        if loss_weights[1] > 0:
            mse_out = output[1]
            mse_target = target[1]
            mse_out[target[0] == 1] = 0
            mse_target[target[0] == 1] = 0
            loss_mse = loss_weights[1] * mse_loss(mse_out.squeeze(1), mse_target.squeeze(1))
            loss_mse_item = loss_mse.item()
            general_loss = general_loss + loss_mse if general_loss else loss_mse

        if loss_weights[2] > 0:
            loss_dvt = loss_weights[2] * mse_loss(output[1], target[1])
            loss_dvt_item = loss_dvt.item()
            general_loss = general_loss + loss_dvt if general_loss else loss_dvt

        return general_loss, loss_bcel_item, loss_mse_item, loss_dvt_item, loss_blur_item
        # return general_loss, 0, 0, loss_dvt

    return custom_loss


def only_mse(loss_weights, window_size, sigma):
    def custom_loss(output, target):
        if output[0].device != target[0].device:
            min_len = min(len(target), len(output))
            for i in range(min_len):  # same processor for comperison
                target[i] = target[i].to(output[i].device)
        mse_loss = nn.MSELoss()
        loss_mse = mse_loss(output[1].squeeze(1), target[1].squeeze(1))
        loss_mse_item = loss_mse.item()
        general_loss = loss_mse
        return general_loss, 0, loss_mse_item, 0, 0

    return custom_loss


def focalbcel_mse_loss(loss_weights, window_size, sigma):
    def custom_loss(output, target):

        if output[0].device != target[0].device:
            min_len = min(len(target), len(output))
            for i in range(min_len):  # same processor for comperison
                target[i] = target[i].to(output[i].device)
        binary_focal_cross_entropy_loss = FocalLossWithLogitsLoss(gamma=sigma).to(output[0].device)
        mse_loss = nn.MSELoss()
        general_loss = None
        loss_bcel_item = 0
        loss_mse_item = 0
        loss_dvt_item = 0
        loss_blur_item = 0

        if loss_weights[0] > 0:
            loss_bcel = binary_focal_cross_entropy_loss(output[0], target[0])  # removing channel dimention
            loss_bcel_item = loss_bcel.item()
            general_loss = loss_weights[0] * loss_bcel

        if loss_weights[1] > 0:
            loss_mse = mse_loss(output[1], target[1])
            loss_mse_item = loss_mse.item()
            general_loss = general_loss + loss_weights[1] * loss_mse if general_loss else loss_weights[1] * loss_mse

        if loss_weights[2] > 0:
            loss_dvt = mse_loss(output[2], target[2])
            loss_dvt_item = loss_dvt.item()
            general_loss = general_loss + loss_weights[2] * loss_dvt if general_loss else loss_weights[2] * loss_dvt

        # if len(loss_weights) > 3 and loss_weights[3] != 0 and False: #cannot be done on a single point
        #     pass
        #     g_blur = SpikeSmoothing(1, window_size, sigma, 1).to('cuda', torch.double)
        #     blur_output = (output[0] >= 0.5).double()
        #     loss_blur = loss_weights[3] * mse_loss(g_blur(blur_output.squeeze(3)), g_blur(target[0].squeeze(3)))
        #     general_loss = general_loss + loss_blur if general_loss else loss_blur
        return general_loss, loss_bcel_item, loss_mse_item, loss_dvt_item, loss_blur_item
        # return general_loss, 0, 0, loss_dvt

    return custom_loss


def focalbcel_mse_mae_loss(loss_weights, window_size, sigma):
    def custom_loss(output, target):

        if output[0].device != target[0].device:
            min_len = min(len(target), len(output))
            for i in range(min_len):  # same processor for comperison
                target[i] = target[i].to(output[i].device)
        binary_focal_cross_entropy_loss = FocalLossWithLogitsLoss(gamma=sigma).to(output[0].device)
        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        general_loss = None
        loss_bcel_item = 0
        loss_mse_item = 0
        loss_mae_item = 0
        loss_blur_item = 0

        if loss_weights[0] > 0:
            loss_bcel = binary_focal_cross_entropy_loss(output[0], target[0])  # removing channel dimention
            loss_bcel_item = loss_bcel.item()
            general_loss = loss_weights[0] * loss_bcel

        if loss_weights[1] > 0:
            loss_mse = mse_loss(output[1], target[1])
            loss_mse_item = loss_mse.item()
            general_loss = general_loss + loss_weights[1] * loss_mse if general_loss else loss_weights[1] * loss_mse

        if loss_weights[2] > 0:
            loss_mae = mae_loss(output[1], target[1])
            loss_mae_item = loss_mae.item()
            general_loss = general_loss + loss_weights[2] * loss_mae if general_loss else loss_weights[2] * loss_mae

        # if len(loss_weights) > 3 and loss_weights[3] != 0 and False: #cannot be done on a single point
        #     pass
        #     g_blur = SpikeSmoothing(1, window_size, sigma, 1).to('cuda', torch.double)
        #     blur_output = (output[0] >= 0.5).double()
        #     loss_blur = loss_weights[3] * mse_loss(g_blur(blur_output.squeeze(3)), g_blur(target[0].squeeze(3)))
        #     general_loss = general_loss + loss_blur if general_loss else loss_blur
        return general_loss, loss_bcel_item, loss_mse_item, loss_mae_item, loss_blur_item
        # return general_loss, 0, 0, loss_dvt

    return custom_loss

from gaussian_smoothing import GaussianSmoothing
import torch.nn as nn
from spike_blur import SpikeSmoothing
import torch

def bcel_mse_dvt_blur_loss(loss_weights, window_size, sigma):
    def custom_loss(output, target):

        if output[0].device != target[0].device:
            min_len = min(len(target),len(output))
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
            loss_mse = loss_weights[1] * mse_loss(output[1].squeeze(1), target[1].squeeze(1))
            loss_mse_item = loss_mse.item()
            general_loss = general_loss + loss_mse if general_loss else loss_mse

        if loss_weights[2] > 0:
            loss_dvt = loss_weights[2] * mse_loss(output[2], target[2])
            loss_dvt_item = loss_dvt.item()
            general_loss = general_loss + loss_dvt if general_loss else loss_dvt

        if len(loss_weights) > 3 and loss_weights[3] != 0 and False: #cannot be done on a single point
            pass
            g_blur = SpikeSmoothing(1, window_size, sigma, 1).to('cuda', torch.double)
            blur_output = (output[0] >= 0.5).double()
            loss_blur = loss_weights[3] * mse_loss(g_blur(blur_output.squeeze(3)), g_blur(target[0].squeeze(3)))
            general_loss = general_loss + loss_blur if general_loss else loss_blur
        return general_loss, loss_bcel_item, loss_mse_item,loss_dvt_item, loss_blur_item
        # return general_loss, 0, 0, loss_dvt

    return custom_loss



from gaussian_smoothing import GaussianSmoothing
import torch.nn as nn
from spike_blur import SpikeSmoothing
def bcel_mse_dvt_blur_loss(loss_weights, window_size, sigma, has_dvt=False):

    def custom_loss(output, target):

        if output[0].device != target[0].device:
            for i in range(len(target) - 1 + has_dvt):  # same processor for comperison
                target[i] = target[i].to(output[i].device)
        binary_cross_entropy_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()
        general_loss = 0
        loss_bcel = loss_weights[0] * binary_cross_entropy_loss(output[0],
                                                                target[0])  # removing channel dimention
        if len(loss_weights) > 3:
            g_blur = GaussianSmoothing(1, window_size, sigma, 1).to('cuda', torch.double)
            loss_blur = loss_weights[3] * mse_loss(g_blur(output[0].squeeze(3)), g_blur(target[0].squeeze(3)))
        loss_mse = loss_weights[1] * mse_loss(output[1].squeeze(1), target[1].squeeze(1))
        loss_dvt = 0
        if has_dvt:
            loss_dvt = loss_weights[2] * mse_loss(output[2], target[2])
            general_loss = loss_bcel + loss_mse + loss_dvt
            return general_loss, loss_bcel.item(), loss_mse.item(), loss_dvt.item()
        loss_blur_val = 0
        if len(loss_weights) > 3:
            general_loss += loss_blur
            loss_blur_val = loss_blur.item()
        general_loss = loss_bcel + loss_mse
        return general_loss, loss_bcel.item(), loss_mse.item(), loss_dvt, loss_blur_val
        # return general_loss, 0, 0, loss_dvt

    return custom_loss

def mse_spike_and_voltage_weighted_loss(loss_weights, window_size, sigma):
    def custom_loss(output, target):
        mse = nn.MSELoss()
        v_loss = loss_weights[1]*mse(output,target)
        s_blur = SpikeSmoothing(1, window_size, sigma, 1).to('cuda', torch.double)
        s_loss = loss_weights[0]*mse(s_blur(output),s_blur(target))
        general_loss = s_loss+v_loss
        return general_loss,0,v_loss.item(),0,s_blur.item()



import torch.nn as nn
from trash.spike_blur import SpikeSmoothing
import torch

def bcel_mse_dvt_loss(loss_weights, window_size, sigma):
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
            loss_bcel =  binary_cross_entropy_loss(output[0],target[0])  # removing channel dimention
            loss_bcel_item = loss_bcel.item()
            general_loss = loss_weights[0] *loss_bcel

        if loss_weights[1] > 0:
            loss_mse =  mse_loss(output[1].squeeze(1), target[1].squeeze(1))
            loss_mse_item = loss_mse.item()
            general_loss = general_loss + loss_weights[1] *loss_mse if general_loss else loss_weights[1] *loss_mse

        if loss_weights[2] > 0:
            loss_dvt =  mse_loss(output[2], target[2])
            loss_dvt_item = loss_dvt.item()
            general_loss = general_loss + loss_weights[2] *loss_dvt if general_loss else loss_weights[2] *loss_dvt

        # if len(loss_weights) > 3 and loss_weights[3] != 0 and False: #cannot be done on a single point
        #     pass
        #     g_blur = SpikeSmoothing(1, window_size, sigma, 1).to('cuda', torch.double)
        #     blur_output = (output[0] >= 0.5).double()
        #     loss_blur = loss_weights[3] * mse_loss(g_blur(blur_output.squeeze(3)), g_blur(target[0].squeeze(3)))
        #     general_loss = general_loss + loss_blur if general_loss else loss_blur
        return general_loss, loss_bcel_item, loss_mse_item,loss_dvt_item, loss_blur_item
        # return general_loss, 0, 0, loss_dvt

    return custom_loss

def loss_zero_mse_on_spikes(loss_weights, window_size, sigma):
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
            mse_out = output[1]
            mse_target = target[1]
            mse_out[target[0]==1]=0
            mse_target[target[0]==1]=0
            loss_mse = loss_weights[1] * mse_loss(mse_out.squeeze(1), mse_target.squeeze(1))
            loss_mse_item = loss_mse.item()
            general_loss = general_loss + loss_mse if general_loss else loss_mse

        if loss_weights[2] > 0:
            loss_dvt = loss_weights[2] * mse_loss(output[2], target[2])
            loss_dvt_item = loss_dvt.item()
            general_loss = general_loss + loss_dvt if general_loss else loss_dvt

        return general_loss, loss_bcel_item, loss_mse_item,loss_dvt_item, loss_blur_item
        # return general_loss, 0, 0, loss_dvt

    return custom_loss

def only_mse(loss_weights, window_size, sigma):
    def custom_loss(output, target):
        loss_mse = mse_loss(output[1].squeeze(1), target[1].squeeze(1))
        loss_mse_item = loss_mse.item()
        general_loss = loss_mse
        return general_loss,0,loss_mse_item,0
    return custom_loss
from typing import Generator, Tuple
import numpy as np

include_DVT=False
def learning_parameters_iter(config) -> Generator[Tuple[int, float, Tuple[float, float, float]], None, None]:
    sigma = config.time_domain_shape
    DVT_loss_mult_factor = 0
    # epoch_in_each_step = config.num_epochs // 5 + (config.num_epochs % 5 != 0)
    while(True):
        learning_rate_per_epoch = 1. / ((np.sqrt(config.epoch_counter) + 1) * 10000)
        loss_weights_per_epoch = [1.0, 1 / (np.sqrt(config.epoch_counter + 1)), DVT_loss_mult_factor * 0.00005]
        sigma = (sigma-1) / (np.sqrt(config.epoch_counter + 1)) +1
        yield learning_rate_per_epoch, loss_weights_per_epoch, sigma

def learning_parameters_iter_slow_10(config) -> Generator[Tuple[int, float, Tuple[float, float, float]], None, None]:
    sigma = config.time_domain_shape
    DVT_loss_mult_factor = 0
    # epoch_in_each_step = config.num_epochs // 5 + (config.num_epochs % 5 != 0)
    while(True):
        learning_rate_per_epoch = 1. / ((np.sqrt(config.epoch_counter//10) + 1) * 10000)
        loss_weights_per_epoch = [1.0, 10 / (np.sqrt(config.epoch_counter//20) + 1), DVT_loss_mult_factor * 0.00005]
        sigma = (sigma-1) / (np.sqrt(config.epoch_counter//10) + 1) +1
        yield learning_rate_per_epoch, loss_weights_per_epoch, sigma


def learning_parameters_iter_slow_10_with_constant_weights(config) -> Generator[Tuple[int, float, Tuple[float, float, float]], None, None]:
    sigma = config.time_domain_shape
    DVT_loss_mult_factor = 0
    # epoch_in_each_step = config.num_epochs // 5 + (config.num_epochs % 5 != 0)
    while(True):
        learning_rate_per_epoch = 1. / ((np.sqrt(config.epoch_counter//10) + 1) * 10000)
        loss_weights_per_epoch = [10, 1, DVT_loss_mult_factor * 0.00005]
        sigma = (sigma-1) / (np.sqrt(config.epoch_counter//10) + 1) +1
        yield learning_rate_per_epoch, loss_weights_per_epoch, sigma

def learning_parameters_iter_slow_50_with_constant_weights(config) -> Generator[Tuple[int, float, Tuple[float, float, float]], None, None]:
    sigma = config.time_domain_shape
    DVT_loss_mult_factor = 0
    # epoch_in_each_step = config.num_epochs // 5 + (config.num_epochs % 5 != 0)
    while(True):
        learning_rate_per_epoch = 1. / ((np.sqrt(config.epoch_counter//50) + 1) * 10000)
        loss_weights_per_epoch = [10, 1, DVT_loss_mult_factor * 0.00005]
        sigma = (sigma-1) / (np.sqrt(config.epoch_counter//10) + 1) +1
        yield learning_rate_per_epoch, loss_weights_per_epoch, sigma

def learning_parameters_iter_slow_10_with_slow_lr(config) -> Generator[Tuple[int, float, Tuple[float, float, float]], None, None]:
    sigma = config.time_domain_shape
    DVT_loss_mult_factor = 0
    # epoch_in_each_step = config.num_epochs // 5 + (config.num_epochs % 5 != 0)
    while(True):
        learning_rate_per_epoch = 1. / ((np.log(config.epoch_counter//10) + 1) * 1000)
        loss_weights_per_epoch = [1.0, 10 / (np.sqrt(config.epoch_counter//20) + 1), DVT_loss_mult_factor * 0.00005]
        sigma = (sigma-1) / (np.sqrt(config.epoch_counter//10) + 1) +1
        yield learning_rate_per_epoch, loss_weights_per_epoch, sigma

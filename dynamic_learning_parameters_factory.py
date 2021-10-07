from typing import Generator, Tuple
import numpy as np

include_DVT=False
def learning_parameters_iter(config) -> Generator[Tuple[int, float, Tuple[float, float, float]], None, None]:
    sigma = 100
    DVT_loss_mult_factor = 1
    # epoch_in_each_step = config.num_epochs // 5 + (config.num_epochs % 5 != 0)
    while(True):
        learning_rate_per_epoch = 1. / ((np.sqrt(config.epoch_counter) + 1) * 10000)
        loss_weights_per_epoch = [1.0, 1 / (np.sqrt(config.epoch_counter + 1)), DVT_loss_mult_factor * 0.00005]
        sigma = sigma / (config.epoch_counter + 1)
        yield learning_rate_per_epoch, loss_weights_per_epoch, sigma

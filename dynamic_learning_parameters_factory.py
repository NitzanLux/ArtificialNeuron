from typing import Generator, Tuple
include_DVT=False
def learning_parameters_iter(config) -> Generator[Tuple[int, float, Tuple[float, float, float]], None, None]:
    DVT_loss_mult_factor = 0.1
    sigma = 100
    DVT_loss_mult_factor = 1
    epoch_in_each_step = config.num_epochs // 5 + (config.num_epochs % 5 != 0)
    for i in range(epoch_in_each_step):
        learning_rate_per_epoch = 1./((config.epoch_counter+1) * 10000)
        loss_weights_per_epoch = [1.0, 1/((config.epoch_counter + 1)), DVT_loss_mult_factor * 0.00005]
        yield learning_rate_per_epoch, loss_weights_per_epoch, sigma / (config.epoch_counter + 1)
    for i in range(epoch_in_each_step):
        learning_rate_per_epoch = 1./((config.epoch_counter+1) * 10000)
        loss_weights_per_epoch = [1.0, 1/((config.epoch_counter + 1)), DVT_loss_mult_factor * 0.00003]
        yield learning_rate_per_epoch, loss_weights_per_epoch, sigma / (config.epoch_counter + 1)
    for i in range(epoch_in_each_step):
        learning_rate_per_epoch = 1./((config.epoch_counter+1) * 10000)
        loss_weights_per_epoch = [1.0, 1/((config.epoch_counter + 1)), DVT_loss_mult_factor * 0.00001]
        yield learning_rate_per_epoch, loss_weights_per_epoch, sigma / (config.epoch_counter + 1)

    for i in range(config.num_epochs // 5):
        learning_rate_per_epoch = 1./((config.epoch_counter+1) * 10000)
        loss_weights_per_epoch = [1.0, 1/((config.epoch_counter + 1)), DVT_loss_mult_factor * 0.0000001]
        yield learning_rate_per_epoch, loss_weights_per_epoch, sigma / (config.epoch_counter + 1)

    for i in range(config.num_epochs // 5 + config.num_epochs % 5):
        learning_rate_per_epoch = 1./((config.epoch_counter+1) * 10000)
        loss_weights_per_epoch = [1.0, 1/((config.epoch_counter + 1)), DVT_loss_mult_factor * 0.00000001]
        yield learning_rate_per_epoch, loss_weights_per_epoch, sigma / (config.epoch_counter + 1)
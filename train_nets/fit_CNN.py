import argparse
from copy import copy
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model_evaluation import ModelEvaluator
import configuration_factory
import wandb
from utils.general_aid_function import *
from neuron_network import davids_network
from neuron_network import fully_connected_temporal_seperated
from neuron_network import neuronal_model
from neuron_network.node_network import recursive_neuronal_model
from parameters_factories import dynamic_learning_parameters_factory as dlpf, loss_function_factory
from project_path import *
from neuron_simulations.simulation_data_generator_new import *
from datetime import datetime
from general_variables import *
import gc

torch.cuda.empty_cache()

print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
print("done")

WANDB_API_KEY = "2725e59f8f4484605300fdf4da4c270ff0fe44a3"

WANDB_PROJECT_NAME = "ArtificialNeuron1"

DOCUMENT_ON_WANDB = True
WATCH_MODEL = False

NUMBER_OF_HOURS_FOR_PLOTTING_EVALUATIONS_PLOTS = 6000
NUMBER_OF_HOURS_FOR_SAVING_MODEL_AND_CONFIG = 1
VALIDATION_EVALUATION_FREQUENCY = 100
ACCURACY_EVALUATION_FREQUENCY = 100
BATCH_LOG_UPDATE_FREQ = 100
# BUFFER_SIZE_IN_FILES_VALID = 4
BUFFER_SIZE_IN_FILES_VALID = 10
# BUFFER_SIZE_IN_FILES_TRAINING = 8
BUFFER_SIZE_IN_FILES_TRAINING = 100
SAMPLE_RATIO_TO_SHUFFLE_TRAINING = 1.0001

synapse_type = 'NMDA'
include_DVT = False
print_logs=False
# for dibugging
# print_logs=True
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# NUMBER_OF_HOURS_FOR_SAVING_MODEL_AND_CONFIG=0
# BATCH_LOG_UPDATE_FREQ = 1
# VALIDATION_EVALUATION_FREQUENCY=4
# ACCURACY_EVALUATION_FREQUENCY = 4
# BUFFER_SIZE_IN_FILES_VALID = 1
# BUFFER_SIZE_IN_FILES_TRAINING = 1
# DOCUMENT_ON_WANDB = False
print('-----------------------------------------------')
print('finding data')
print('-----------------------------------------------', flush=True)


# ------------------------------------------------------------------
# basic configurations and directories
# ------------------------------------------------------------------


# num_DVT_components = 20 if synapse_type == 'NMDA' else 30


def batch_train(model, optimizer, custom_loss, train_data_iterator, clip_gradient, accumulate_loss_batch_factor,
                optimizer_scdualer, scaler):
    # zero the parameter gradients
    torch.cuda.empty_cache()
    gc.collect()
    model.train()
    general_loss, loss_bcel, loss_mse, loss_dvt, loss_gausian_mse = 0, 0, 0, 0, 0
    if scaler is None:
        out = train_without_mixed_precision(accumulate_loss_batch_factor, clip_gradient, custom_loss, general_loss,
                                            loss_bcel,
                                            loss_dvt, loss_gausian_mse, loss_mse, model, optimizer, optimizer_scdualer,
                                            train_data_iterator)
    else:
        out = train_with_mixed_precision(accumulate_loss_batch_factor, clip_gradient, custom_loss, general_loss,
                                         loss_bcel,
                                         loss_dvt, loss_gausian_mse, loss_mse, model, optimizer, optimizer_scdualer,
                                         scaler,
                                         train_data_iterator)

    return out


def train_without_mixed_precision(accumulate_loss_batch_factor, clip_gradient, custom_loss, general_loss, loss_bcel,
                                  loss_dvt, loss_gausian_mse, loss_mse, model, optimizer, optimizer_scdualer,
                                  train_data_iterator):
    for _, data in zip(range(accumulate_loss_batch_factor), train_data_iterator):
        torch.cuda.empty_cache()
        inputs, labels = data
        inputs = inputs.cuda().type(DATA_TYPE)
        labels = [l.cuda().flatten().type(DATA_TYPE) for l in labels]
        # forward + backward + optimize
        outputs = model(inputs)
        outputs = [i.flatten() for i in outputs]
        cur_general_loss, cur_loss_bcel, cur_loss_mse, cur_loss_dvt, cur_loss_gausian_mse = custom_loss(outputs, labels)
        cur_general_loss /= accumulate_loss_batch_factor
        cur_general_loss.backward()
        general_loss += cur_general_loss
        loss_bcel += cur_loss_bcel / accumulate_loss_batch_factor
        loss_mse += cur_loss_mse / accumulate_loss_batch_factor
        loss_dvt += cur_loss_dvt / accumulate_loss_batch_factor
        loss_gausian_mse += cur_loss_gausian_mse / accumulate_loss_batch_factor

    if clip_gradient is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
    if optimizer_scdualer is not None:
        optimizer_scdualer.step(general_loss)
    optimizer.step()
    optimizer.zero_grad()
    out = general_loss, loss_bcel, loss_mse, loss_dvt, loss_gausian_mse
    return out


def train_with_mixed_precision(accumulate_loss_batch_factor, clip_gradient, custom_loss, general_loss, loss_bcel,
                               loss_dvt, loss_gausian_mse, loss_mse, model, optimizer, optimizer_scdualer, scaler,
                               train_data_iterator):
    for _, data in zip(range(accumulate_loss_batch_factor), train_data_iterator):
        inputs, labels = data
        inputs = inputs.cuda().type(DATA_TYPE)
        with torch.cuda.amp.autocast():
            labels = [l.cuda().flatten().type(DATA_TYPE) for l in labels]
            # forward + backward + optimize
            outputs = model(inputs)
            outputs = [i.flatten() for i in outputs]
            cur_general_loss, cur_loss_bcel, cur_loss_mse, cur_loss_dvt, cur_loss_gausian_mse = custom_loss(outputs,
                                                                                                            labels)
            scaler.scale(cur_general_loss / accumulate_loss_batch_factor).backward()
            general_loss += cur_general_loss / accumulate_loss_batch_factor
            loss_bcel += cur_loss_bcel / accumulate_loss_batch_factor
            loss_mse += cur_loss_mse / accumulate_loss_batch_factor
            loss_dvt += cur_loss_dvt / accumulate_loss_batch_factor
            loss_gausian_mse += cur_loss_gausian_mse / accumulate_loss_batch_factor

        # plot_grad_flow(model)
        # plt.show()
        # for n,p in model.named_parameters():
        #     if 'weight' in n:
        #         print('===========\ngradient:{}\n----------\n{}\n*\n{}'.format(n, p.grad.mean(),p.grad.mean()))
        # unscaling and clipping
    scaler.unscale_(optimizer)
    if clip_gradient is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
    if optimizer_scdualer is not None:
        optimizer_scdualer.step(general_loss)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    out = general_loss, loss_bcel, loss_mse, loss_dvt, loss_gausian_mse
    return out


def plot_grad_flow(model=None):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    model.cuda().eval()
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in model.named_parameters():
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().numpy())
            max_grads.append(p.grad.abs().max().cpu().numpy())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.show()


def train_network(config, model):
    DVT_PCA_model = None

    model.cuda().train()
    if DATA_TYPE == torch.cuda.FloatTensor:
        model.float()
    elif DATA_TYPE == torch.cuda.DoubleTensor:
        model.double()
    train_data_generator, validation_data_generator = get_data_generators(config)
    validation_data_iterator = iter(validation_data_generator)
    batch_counter = 0
    saving_counter = 0
    optimizer_scheduler = None
    custom_loss = None
    optimizer = None
    evaluation_plotter_scheduler = SavingAndEvaluationScheduler()
    if not config.dynamic_learning_params:
        learning_rate, loss_weights, optimizer, sigma, custom_loss = generate_constant_learning_parameters(config,
                                                                                                           model)
        if config.lr_scheduler is not None:
            optimizer_scheduler = getattr(lr_scheduler, config.lr_scheduler)(optimizer, **config.lr_scheduler_params)
        dynamic_parameter_loss_genrator = None
    else:
        learning_rate, loss_weights, sigma = 0.001, [1] * 3, 0.1  # default values
        dynamic_parameter_loss_genrator = getattr(dlpf, config.dynamic_learning_params_function)(config)

    scaler = torch.cuda.amp.GradScaler(enabled=True) if config.use_mixed_precision else None
    if DOCUMENT_ON_WANDB and WATCH_MODEL:
        wandb.watch(model, log='all', log_freq=1, log_graph=True)
    model_level_training_scadualer=None
    if isinstance(model, recursive_neuronal_model.RecursiveNeuronModel):
        model_level_training_scadualer = model.train_random_subtree(
        config.freeze_node_factor if config.freeze_node_factor is not None else 0)
    print("start training...", flush=True)
    for epoch in range(config.num_epochs):
        config.update(dict(epoch_counter=config.epoch_counter + 1), allow_val_change=True)
        # epoch_start_time = time.time()
        loss_weights, optimizer, sigma, custom_loss = set_dynamic_learning_parameters(config,
                                                                                      dynamic_parameter_loss_genrator,
                                                                                      loss_weights, model, optimizer,
                                                                                      custom_loss, sigma)
        train_data_iterator = iter(train_data_generator)
        if model_level_training_scadualer is not None:
            next(model_level_training_scadualer)

        for i in range(config.epoch_size):
            saving_counter += 1
            if DOCUMENT_ON_WANDB:
                wandb.log({}, commit=True)
            config.update(dict(batch_counter=config.batch_counter + 1), allow_val_change=True)
            # get the inputs; data is a list of [inputs, labels]

            train_loss = batch_train(model, optimizer, custom_loss, train_data_iterator, config.clip_gradients_factor,
                                     config.accumulate_loss_batch_factor, optimizer_scheduler, scaler)
            lr = log_lr(config, optimizer)
            train_log(train_loss, config.batch_counter, epoch, lr, sigma, loss_weights, additional_str="train")
            evaluate_validation(config, custom_loss, model, validation_data_iterator)
            # save model every once a while
            if saving_counter % 10 == 0:
                evaluation_plotter_scheduler(model, config)


def load_model(config):
    print("loading model...", flush=True)
    if config.architecture_type == "DavidsNeuronNetwork":
        model = davids_network.DavidsNeuronNetwork.load(config)
    elif config.architecture_type == "FullNeuronNetwork":
        model =fully_connected_temporal_seperated.FullNeuronNetwork.load(config)
    elif "network_architecture_structure" in config and config.network_architecture_structure == "recursive":
        model = recursive_neuronal_model.RecursiveNeuronModel.load(config)
    else:
        model = neuronal_model.NeuronConvNet.build_model_from_config(config)
    if config.batch_counter == 0:
        model.init_weights(config.init_weights_sd)
    print("model parmeters: %d" % model.count_parameters())
    return model


def log_lr(config, optimizer):
    lr = float(optimizer.param_groups[0]['lr'])
    if lr != config.optimizer_params['lr']:
        optim_params = config.optimizer_params
        optim_params['lr'] = lr
        config.update(dict(optim_params=optim_params), allow_val_change=True)
    return lr


def set_dynamic_learning_parameters(config, dynamic_parameter_loss_genrator, loss_weights, model, optimizer,
                                    custom_loss, sigma):
    if config.dynamic_learning_params:
        learning_rate, loss_weights, sigma = next(dynamic_parameter_loss_genrator)
        if "loss_function" in config:
            custom_loss = getattr(loss_function_factory, config.loss_function)(loss_weights,
                                                                               config.time_domain_shape, sigma)
        else:
            custom_loss = loss_function_factory.bcel_mse_dvt_loss(loss_weights, config.time_domain_shape, sigma)
        config.optimizer_params["lr"] = float(learning_rate)
        optimizer = getattr(optim, config.optimizer_type)(model.parameters(),
                                                          **config.optimizer_params)
    return loss_weights, optimizer, sigma, custom_loss


def evaluate_validation(config, custom_loss, model, validation_data_iterator):
    if not (
            config.batch_counter % VALIDATION_EVALUATION_FREQUENCY == 0 or config.batch_counter % ACCURACY_EVALUATION_FREQUENCY == 0 or config.batch_counter % ACCURACY_EVALUATION_FREQUENCY == 0):
        return
    if print_logs: print("validate %d" % config.batch_counter)
    valid_input, valid_labels = next(validation_data_iterator)
    model.eval()
    valid_input = valid_input.cuda().type(DATA_TYPE)
    valid_labels = [l.cuda().type(DATA_TYPE) for l in valid_labels]
    with torch.no_grad():
        output = model(valid_input)
    target_s = valid_labels[0].detach().cpu()
    target_s=target_s
    target_s = target_s.numpy()

    target_s=target_s.astype(bool).squeeze().flatten()
    if config.include_spikes:
        output_s = torch.nn.Sigmoid()(output[0])
        output_s = output_s.detach().cpu().numpy().squeeze().flatten()
    else:
        output_s = torch.nn.Sigmoid()(output[1] + 50)
        output_s = output_s.detach().cpu().numpy().squeeze().flatten()
    if config.batch_counter % VALIDATION_EVALUATION_FREQUENCY == 0 or config.batch_counter % ACCURACY_EVALUATION_FREQUENCY == 0:
        validation_loss = custom_loss(output, valid_labels)
        train_log(validation_loss, config.batch_counter, None,
                  additional_str="validation", commit=False)

        target_v = valid_labels[1].detach().cpu().numpy().squeeze().flatten()
        output_v = output[1].detach().cpu().numpy().squeeze().flatten()
        log_dict = {"brier score(s) validation": skm.brier_score_loss(target_s, output_s),
                    "R squared score(v) validation": skm.r2_score(target_v, output_v)}
        if DOCUMENT_ON_WANDB:
            wandb.log(log_dict, step=config.batch_counter, commit=False)  # add training parameters per step

    if config.batch_counter % ACCURACY_EVALUATION_FREQUENCY == 0:
        display_accuracy(target_s, output_s, config.batch_counter,
                         additional_str="validation")
    model.train()


def get_data_generators( config):
    print("loading data...training", flush=True)
    prediction_length = 1
    if config.config_version >= 1.2:
        prediction_length = config.prediction_length
    train_files, valid_files, test_files = load_files_names(config.data_base_path,config.files_filter_regex)
    train_data_generator = SimulationDataGenerator(train_files, buffer_size_in_files=BUFFER_SIZE_IN_FILES_TRAINING,
                                                   prediction_length=prediction_length,
                                                   batch_size=config.batch_size_train,
                                                   epoch_size=config.epoch_size * config.accumulate_loss_batch_factor,
                                                   window_size_ms=config.time_domain_shape,
                                                   sample_ratio_to_shuffle=SAMPLE_RATIO_TO_SHUFFLE_TRAINING,
                                                   )
    print("loading data...validation", flush=True)

    validation_data_generator = SimulationDataGenerator(train_files,#valid_files,
                                                        buffer_size_in_files=BUFFER_SIZE_IN_FILES_VALID,
                                                        prediction_length=5780,
                                                        batch_size=config.batch_size_validation,
                                                        window_size_ms=config.time_domain_shape,
                                                        sample_ratio_to_shuffle=1.5,
                                                        )
    if "spike_probability" in config and config.spike_probability is not None:
        train_data_generator.change_spike_probability(config.spike_probability)
    # validation_data_generator.change_spike_probability(0.5)
    print("finished with the data!!!", flush=True)

    return train_data_generator, validation_data_generator


class SavingAndEvaluationScheduler():
    """
    create evaluation plots , every 23 hours.
    :return: evaluation
    """

    def __init__(self, time_in_hours_for_saving=NUMBER_OF_HOURS_FOR_SAVING_MODEL_AND_CONFIG,
                 time_in_hours_for_evaluation=NUMBER_OF_HOURS_FOR_PLOTTING_EVALUATIONS_PLOTS):
        self.last_time_evaluation = datetime.now()
        self.last_time_saving = datetime.now()
        self.time_in_hours_for_saving = time_in_hours_for_saving
        self.time_in_hours_for_evaluation = time_in_hours_for_evaluation

    def create_evaluation_schduler(self, config, model=None):
        current_time = datetime.now()
        delta_time = current_time - self.last_time_evaluation
        if (delta_time.total_seconds() / 60) / 60 > self.time_in_hours_for_evaluation:
            ModelEvaluator.build_and_save(config=config, model=model)
            self.last_time_evaluation = datetime.now()

    def save_model_schduler(self, config, model):
        current_time = datetime.now()
        delta_time = current_time - self.last_time_saving
        if (delta_time.total_seconds() / 60) / 60 > self.time_in_hours_for_saving:
            self.save_model(model, config)
            self.last_time_saving = datetime.now()

    @staticmethod
    def flush_all(config, model):
        SavingAndEvaluationScheduler.save_model(model, config)
        # ModelEvaluator.build_and_save(config=config, model=model)

    @staticmethod
    def save_model(model, config: AttrDict):
        print('-----------------------------------------------------------------------------------------')
        print('finished epoch %d saving...\n     "%s"\n"' % (
            config.epoch_counter, config.model_filename.split('/')[-1]))
        print('-----------------------------------------------------------------------------------------')
        #backup
        base_path=os.path.dirname(os.path.join(MODELS_DIR, *config.model_path))
        files_path=os.listdir(base_path)
        for fn in files_path:
            fn = str(fn)
            filename, file_extension = os.path.splitext(fn)
            if 'temp' not in file_extension:
                if os.path.exists(os.path.join(base_path,fn)+'temp'):
                    os.remove(os.path.join(base_path,fn)+'temp')

                os.rename(os.path.join(base_path, fn), os.path.join(base_path, fn) + 'temp')
        model.save(os.path.join(MODELS_DIR, *config.model_path))
        configuration_factory.overwrite_config(AttrDict(config))

    def __call__(self, model, config):
        self.create_evaluation_schduler(config, model)
        self.save_model_schduler(config, model)


def generate_constant_learning_parameters(config, model):
    loss_weights = config.constant_loss_weights
    sigma = config.constant_sigma
    learning_rate = None
    if "loss_function" in config:
        custom_loss = getattr(loss_function_factory, config["loss_function"])(loss_weights,
                                                                              config.time_domain_shape, sigma)
    else:
        custom_loss = loss_function_factory.bcel_mse_dvt_loss(loss_weights, config.time_domain_shape, sigma)
    if "lr" in (config.optimizer_params):
        config.update(dict(constant_learning_rate=float(config.optimizer_params["lr"])), allow_val_change=True)
    else:
        optimizer_params = copy(config.optimizer_params)
        optimizer_params["lr"] = float(config.constant_learning_rate)
        config.update(dict(optimizer_params=optimizer_params), allow_val_change=True)

    optimizer = getattr(optim, config.optimizer_type)(model.parameters(),
                                                      **config.optimizer_params)
    return learning_rate, loss_weights, optimizer, sigma, custom_loss


def model_pipline(hyperparameters):
    if DOCUMENT_ON_WANDB:
        wandb.login()
        with wandb.init(project=(WANDB_PROJECT_NAME), config=hyperparameters, entity='nilu', allow_val_change=True,
                        id=hyperparameters.model_filename):
            config = wandb.config
            load_and_train(config)
    else:
        config = hyperparameters
        load_and_train(config)


def load_and_train(config):
    model = load_model(config)
    try:
        train_network(config, model)
    finally:
        # pass
        SavingAndEvaluationScheduler.flush_all(config, model)


def train_log(loss, step, epoch=None, learning_rate=None, sigma=None, weights=None, additional_str='', commit=False):
    general_loss, loss_bcel, loss_mse, loss_dvt, blur_loss = loss
    general_loss = float(general_loss.item())
    if print_logs: print("train", flush=True)

    if DOCUMENT_ON_WANDB:
        log_dict = {"general loss %s" % additional_str: general_loss,
                    "mse loss %s" % additional_str: loss_mse, "bcel loss %s" % additional_str: loss_bcel,
                    "dvt loss %s" % additional_str: loss_dvt, "blur loss %s" % additional_str: blur_loss}
        if epoch is not None:
            log_dict.update({"epoch": epoch})
        if learning_rate is not None:
            log_dict.update({"learning rate %s" % additional_str: learning_rate})  # add training parameters per step
        if weights is not None:
            log_dict.update({"bcel weight  %s" % additional_str: weights[0],
                             "dvt weight  %s" % additional_str: weights[2],
                             "mse weight  %s" % additional_str: weights[1]})  # add training parameters per step
        if sigma is not None:
            log_dict.update({"sigma %s" % additional_str: sigma})  # add training parameters per step
        wandb.log(log_dict, step=step, commit=commit)  # add training parameters per step
    if print_logs:
        print("step %d, epoch %d %s" % (step, epoch if epoch is not None else -1, additional_str))
        print("general loss ", general_loss)
        print("mse loss ", loss_mse)
        print("bcel loss ", loss_bcel)
        print("dvt loss ", loss_dvt)


def display_accuracy(target, output, step, additional_str=''):
    if not DOCUMENT_ON_WANDB or step == 0:
        return
    output = np.vstack([np.abs(1 - output), output]).T
    # fpr, tpr, thresholds = skm.roc_curve(target, output[:,1], )  # wandb has now possible to extruct it yet
    auc = skm.roc_auc_score(target, output[:, 1])
    if print_logs: print("AUC   ", auc)
    wandb.log({"pr %s" % additional_str: wandb.plot.pr_curve(target, output,
                                                             labels=None, classes_to_plot=None),
               "roc %s" % additional_str: wandb.plot.roc_curve(target, output, labels=None, classes_to_plot=None),
               "AUC %s" % additional_str: auc}, commit=False)
    # if "best_accuracy" in wandb.run.summary:
    #     best_accuracy = wandb.run.summary["best_accuracy"]
    #     if best_accuracy is None: best_accuracy=0
    # else:
    #     best_accuracy = 0
    # wandb.run.summary["best_accuracy"] = max(best_accuracy, auc)
    # wandb.run.summary.update()


def run_fit_cnn():
    global e

    parser = argparse.ArgumentParser(description='Add configuration file')
    parser.add_argument(dest="config_path", type=str,
                        help='configuration file for path')
    parser.add_argument(dest="job_id", help="the job id", type=str)
    args = parser.parse_args()
    print(args)
    config = configuration_factory.load_config_file(args.config_path)
    # set SEED
    torch.manual_seed(int(config.torch_seed))
    # np.random.seed(int(config.numpy_seed))
    # random.seed(int(config.random_seed))
    # try:
    model_pipline(config)
    # configuration_factory.
    # except Exception as e:
    # send_mail("nitzan.luxembourg@mail.huji.ac.il","somthing went wrong",e)
    # raise e


run_fit_cnn()

# send_mail("nitzan.luxembourg@mail.huji.ac.il","finished run","finished run")
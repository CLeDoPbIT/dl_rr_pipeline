import numpy as np
import torch
import torch.nn
import models
import losses
import metrics
import  datasets
from utils.misc import read_json, str_to_bool, save_weights, read_data, load_weights
from torch import optim


def __test(model, data_loader, metric, device):
    """
    Function of test result

    Args:
        model: model for test
        data_loader: data loader for model
        metric: class for score result
        device: gpu or cpu

    Returns:
        score: score of model

    """

    model.eval()
    epoch_labels = list()
    epoch_preds = list()
    for inputs, labels in data_loader:

        inputs = inputs.to(device, non_blocking=True)
        inputs = inputs.float()
        labels = labels.to(device, non_blocking=True, dtype=torch.long)

        output = model(inputs)
        _, preds = torch.max(output, 1)

        epoch_labels = np.concatenate((epoch_labels, labels.cpu().numpy()), axis=0)
        epoch_preds = np.concatenate((epoch_preds, preds.cpu().numpy()), axis=0)

    score = metric(epoch_preds, epoch_labels)

    return score


def __init_model(network_config, best_snapshot):
    """
    Init model for training with respect to config

    Args:
        network_config: config of network and train process
        best_snapshot: best snapshot for testins

    Returns:
        model, device: model for training, gpu or cpu

    """

    model = getattr(models, network_config['arch'])(network_config['input_size'],network_config['number_classes'])

    if bool(network_config["train_on_gpu"]):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            raise KeyError("Try to create model on GPU, but GPU is not available")
    else:
        device = ("cpu")

    model = model.to(device)

    if torch.cuda.is_available() and network_config["parallel"]:
        model = torch.nn.DataParallel(model)
        model = load_weights(model, best_snapshot, parallel=True)
    else:
        model = load_weights(model, best_snapshot, parallel=False)

    return model, device


def __init_parameters_network(network_config):
    """
    Init parameters score metric

    Args:
        network_config: config of network and train process

    Returns:
        criterion, metric, optimizer, lr_scheduler: loss function, score metric, optimizer, learning rate scheduler

    """

    metric = getattr(metrics, network_config['metrics'])()
    return metric


def process(input_data, input_config, output_data_types):
    """
    Main process of test model

    Args:
        input_data: input data for training
            input_data[0]: test data
            input_data[2]: best snapshot path
        input_config: config of network
    """

    x_test, y_test = read_data(input_data[list(input_data.keys())[0]], "x_test", "y_test")
    network_config = read_json(input_config[list(input_config.keys())[0]])
    network_config = str_to_bool(network_config)

    model, device = __init_model(network_config, input_data[list(input_data.keys())[1]])
    metric = __init_parameters_network(network_config)
    dataloader_test = datasets.create_dataloader(network_config, getattr(datasets, network_config["dataset"]),
                                                 x_test, y_test)
    val_score = __test(model, dataloader_test, metric, device)

    print(f"INFO: Test {network_config['arch']}: Accuracy = {val_score}")


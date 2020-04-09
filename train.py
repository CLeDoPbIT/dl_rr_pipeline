import numpy as np
import torch
import torch.nn
import models
import losses
import metrics
import datasets
import os
from utils.misc import read_json, str_to_bool, save_weights, read_data, create_current_date
from torch.utils.tensorboard import SummaryWriter
from torch import optim


def __train(model, data_loader, metric, criterion, optimizer, device):
    """
    Function of train of one epoch

    Args:
        model: model for train
        data_loader: data loader for model
        metric: class for score result
        criterion: loss function
        optimizer: optimizer for model
        device: gpu or cpu

    Returns:
        score, epoch_loss: score of model, epoch loss of model

    """
    model.train()

    epoch_labels = list()
    epoch_preds = list()
    epoch_loss = 0

    # train loop
    for inputs, labels in data_loader:

        inputs = inputs.to(device, non_blocking=True)
        inputs = inputs.float()
        labels = labels.to(device, non_blocking=True, dtype=torch.long)

        output = model(inputs)
        _, preds = torch.max(output, 1)

        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # save results of every batch
        epoch_labels = np.concatenate((epoch_labels, labels.cpu().numpy()), axis = 0)
        epoch_preds = np.concatenate((epoch_preds, preds.cpu().numpy()), axis = 0)
        epoch_loss += loss.item() * inputs.size(0)

    # calculate score
    score = metric(epoch_preds, epoch_labels)
    return score, epoch_loss


def __validation(model, data_loader, metric, criterion, device):
    """
    Function of validation of one epoch

    Args:
        model: model for train
        data_loader: data loader for model
        metric: class for score result
        criterion: loss function
        device: gpu or cpu

    Returns:
        score, epoch_loss: score of model, epoch loss of model

    """

    model.eval()
    epoch_labels = list()
    epoch_preds = list()
    epoch_loss = 0

    # val loop
    for inputs, labels in data_loader:

        inputs = inputs.to(device, non_blocking=True)
        inputs = inputs.float()
        labels = labels.to(device, non_blocking=True, dtype=torch.long)

        output = model(inputs)
        _, preds = torch.max(output, 1)

        loss = criterion(output, labels)

        # save results of every batch
        epoch_labels = np.concatenate((epoch_labels, labels.cpu().numpy()), axis=0)
        epoch_preds = np.concatenate((epoch_preds, preds.cpu().numpy()), axis=0)
        epoch_loss += loss.item() * inputs.size(0)

    # calculate score
    score = metric(epoch_preds, epoch_labels)

    return score, epoch_loss


def __init_model(network_config):
    """
    Init model for training with respect to config

    Args:
        network_config: config of network and train process

    Returns:
        model, device: model for training, gpu or cpu

    """

    # get model
    model = getattr(models, network_config['arch'])(network_config['input_size'][-1],network_config['number_classes'])

    # set train on cpu or gpu
    if bool(network_config["train_on_gpu"]):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            raise KeyError("Try to create model on GPU, but GPU is not available")
    else:
        device = ("cpu")

    model = model.to(device)

    # set train on a few gpu or on one gpu
    if torch.cuda.is_available() and network_config["parallel"]:
        model = torch.nn.DataParallel(model)
    return model, device


def __init_parameters_network(model, network_config):
    """
    Init parameters of loss function,  optimizer, learning rate scheduler and score metric

    Args:
        model: model for parameters
        network_config: config of network and train process

    Returns:
        criterion, metric, optimizer, lr_scheduler: loss function, score metric, optimizer, learning rate scheduler

    """

    criterion = getattr(losses, network_config['loss'])()
    metric = getattr(metrics, network_config['metrics'])()
    optimizer = getattr(optim, network_config['optimizer'])(model.parameters(),
                                                            lr=network_config['learning_rate'])
    lr_scheduler = getattr(optim.lr_scheduler, network_config['lr_scheduler'])(optimizer)
    return criterion, metric, optimizer, lr_scheduler


def __create_network_graph(model, writer, config, device):
    """
    Create network graph for tensorboard

    Args:
        model: model for parameters
        config: config of network
        device: gpu or cpu
        writer: writer of tensorboard

    """

    dummy = torch.rand(config['input_size'])
    dummy = dummy.to(device)
    writer.add_graph(model, input_to_model=dummy)


def __add_current_stat_to_tb(accuracy, loss, writer, phase, epoch):
    """
    Add to tensorboard metrics of current epoch

    Args:
        accuracy: accuracy of model
        loss: config of network
        phase: train or val
        writer: writer of tensorboard
        epoch: current epoch

    """

    writer.add_scalar(phase + "_accuracy",accuracy, global_step=epoch)
    writer.add_scalar(phase + "_loss",loss, global_step=epoch)


def process(input_data, input_config, output_data_types):
    """
    Main process of train model

    Args:
        input_data: input data for training
            input_data[0]: train data
            input_data[1]: val data
            input_data[2]: tensorboard path
        input_config: config of network
        output_data_types: output data:
            output_data_types[0]: best snapshot
    """
    # create current tensorboard exp
    date = create_current_date()
    input_keys = list(input_data.keys())
    writer = SummaryWriter(os.path.join(input_data[input_keys[2]], date))

    # init data and configs
    x_train, y_train = read_data(input_data[input_keys[0]], "x_train", "y_train")
    x_val, y_val = read_data(input_data[input_keys[1]], "x_val", "y_val")
    network_config = read_json(input_config[list(input_config.keys())[0]])

    # convert str boolean to bool
    network_config = str_to_bool(network_config)

    # init model and parameters of training
    model, device = __init_model(network_config)
    criterion, metric, optimizer, lr_scheduler = __init_parameters_network(model, network_config)
    dataloader_train = datasets.create_dataloader(network_config, getattr(datasets, network_config["dataset"]), x_train, y_train)
    dataloader_val = datasets.create_dataloader(network_config, getattr(datasets, network_config["dataset"]), x_val, y_val)

    # save cuurent model graph to tensorboard
    __create_network_graph(model, writer, network_config, device)

    best_loss = np.inf
    start_epoch = 0
    epochs_with_no_progress = 0

    # train val loop
    for epoch in range(start_epoch, network_config["epochs"]):

        train_score, train_loss = __train(model, dataloader_train, metric, criterion, optimizer, device)
        val_score, val_loss = __validation(model, dataloader_val, metric, criterion, device)

        __add_current_stat_to_tb(train_score, train_loss, writer, "train", epoch)
        __add_current_stat_to_tb(val_score, val_loss, writer, "val", epoch)

        print(f"INFO: Train {network_config['arch']:15}, epoch = {epoch:3}: Accuracy = {train_score:8}, Loss = {train_loss:8}")
        print(f"INFO: Val   {network_config['arch']:15}, epoch = {epoch:3}: Accuracy = {val_score:8}, Loss = {val_loss:8}")

        lr_scheduler.step(epoch)

        if best_loss > val_loss:
            best_loss = val_loss
            save_weights(model, output_data_types[list(output_data_types.keys())[0]], parallel=network_config["parallel"])
            save_weights(model, os.path.join(input_data[input_keys[2]], date, output_data_types[list(output_data_types.keys())[0]].split("/")[-1]), parallel=network_config["parallel"])

        else:  # early_stopping
            epochs_with_no_progress += 1
            if epochs_with_no_progress >= network_config["max_epochs_without_progress"]:
                break
    writer.close()

import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from utils.misc import read_json


def process(input_data, input_config, output_data_types):

    raw_test_data_path = input_data["RAW_TRAIN_DATA"]
    raw_train_data_path = input_data["RAW_TEST_DATA"]
    numpy_train_output_path = output_data_types["NUMPY_TRAIN_DATA"]
    numpy_val_output_path = output_data_types["NUMPY_VAL_DATA"]
    numpy_test_output_path = output_data_types["NUMPY_TEST_DATA"]

    numpy_resnet_train_output_path = output_data_types[ "RESNET_NUMPY_TRAIN_DATA"]
    numpy_resnet_val_output_path = output_data_types["RESNET_NUMPY_VAL_DATA"]
    numpy_resnet_test_output_path = output_data_types["RESNET_NUMPY_TEST_DATA"]

    config = read_json(input_config["CREATE_NUMPY_DATA_PROCESSOR_CONFIG"])

    __create_train_val_numpy_files(raw_train_data_path, numpy_train_output_path, numpy_val_output_path,
                                   numpy_resnet_train_output_path, numpy_resnet_val_output_path, config)
    __create_test_numpy_files(raw_test_data_path, numpy_test_output_path, numpy_resnet_test_output_path)


def __create_train_val(x_data, y_data, config):

    # fix random seed for reproduciblity
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data,
                                                        stratify=y_data,
                                                        test_size=config["val_size"],
                                                        random_state=config["seed"])
    return x_train, y_train, x_val, y_val


def __create_train_val_numpy_files(raw_data_path, numpy_train_output_path, numpy_val_output_path,
                                   numpy_resnet_train_output_path, numpy_resnet_val_output_path, config):
    x_data, y_data, x_data_resnet = __create_one_df(raw_data_path)
    x_train, y_train, x_val, y_val = __create_train_val(x_data, y_data, config)
    x_train_resnet, y_train_resnet, x_val_resnet, y_val_resnet = __create_train_val(x_data_resnet, y_data, config)

    np.savez(numpy_train_output_path, x_train=x_train, y_train=y_train)
    np.savez(numpy_val_output_path, x_val=x_val, y_val=y_val)

    np.savez(numpy_resnet_train_output_path, x_train=x_train_resnet, y_train=y_train_resnet)
    np.savez(numpy_resnet_val_output_path, x_val=x_val_resnet, y_val=y_val_resnet)


def __create_test_numpy_files(raw_data_path, numpy_test_output_path, numpy_resnet_test_output_path):
    x_test, y_test, x_test_resnet = __create_one_df(raw_data_path)
    np.savez(numpy_test_output_path, x_test=x_test, y_test=y_test)
    np.savez(numpy_resnet_test_output_path, x_test=x_test_resnet, y_test=y_test)


def __create_one_df(raw_data_path):
    # get all patients
    file_list = os.listdir(os.path.join(raw_data_path))

    # read all the data into a single data frame
    frames = [pd.read_csv(os.path.join(os.path.join(raw_data_path), p), header=None) for p in file_list]
    data = pd.concat(frames, ignore_index=True)

    # split the data into variables and targets (0 = no af, 1 = af)
    x_data = data.iloc[:, 1:].values
    x_data_resnet = x_data.reshape((x_data.shape[0], 1, x_data.shape[1]))
    y_data = data.iloc[:, 0].values
    y_data[y_data > 0] = 1

    return x_data, y_data, x_data_resnet

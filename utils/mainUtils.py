import json
import os


def read_json(filepath):
    config_file = open(filepath, 'r')
    config_json = json.load(config_file)
    config_file.close()
    return config_json


def get_processor_data(data_list, constants_config):
    input_data = dict()
    for data in data_list:
        if data in constants_config:
            input_data[data] = constants_config[data]
        else:
            raise KeyError("There are no data in constants.json")
    return input_data


def is_processor_output_created(processor_output_data_types):
    for data_type in processor_output_data_types:
        if not len(os.listdir(processor_output_data_types[data_type]))==0:
            return True
    return False


def get_processor_output():
    pass


def get_process_input():
    pass


def get_process_data():
    pass


def get_process_tags():
    pass


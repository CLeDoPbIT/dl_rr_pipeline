import os


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
        if os.path.isdir(processor_output_data_types[data_type]):
            if not len(os.listdir(processor_output_data_types[data_type]))==0:
                return True
        else:
            if os.path.exists(processor_output_data_types[data_type]):
                return True
    return False

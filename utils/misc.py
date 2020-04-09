import json


def save_weights():
    pass


def str_to_bool(config_json):
    for key in config_json:
        if config_json[key] == "True":
            config_json[key] = True
        elif config_json[key] == "False":
            config_json[key] = False
    return config_json


def read_json(filepath):
    config_file = open(filepath, 'r')
    config_json = json.load(config_file)
    config_file.close()
    return config_json
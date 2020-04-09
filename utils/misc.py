import json
import pytz
from datetime import datetime
import numpy as np
import torch


def create_current_date():
    utc_now = pytz.utc.localize(datetime.utcnow())
    pst_now = utc_now.astimezone(pytz.timezone("Europe/Moscow"))
    now = datetime.now(tz=pst_now.tzinfo)
    date = str(now)[:19]
    list_date = list(date)
    list_date[10] = '_'
    [list_date.remove("-") for i in range(2)]
    [list_date.remove(":") for i in range(2)]
    date = ''.join(list_date)
    return date


def load_weights(model, file, parallel):
    checkpoint = torch.load(file)
    if torch.cuda.is_available() and parallel:
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model


def save_weights(model, file, parallel):
    if torch.cuda.is_available() and parallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'state_dict': state_dict}, file)


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


def read_data(data_path, x_name, y_name):
    data = np.load(data_path)
    return data[x_name], data[y_name]
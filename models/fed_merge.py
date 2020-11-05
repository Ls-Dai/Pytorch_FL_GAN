import copy
import torch
import torch.nn as nn


def fedavg(parameters):
    parameters_avg = copy.deepcopy(parameters[0])
    for key in parameters_avg.keys():
        for i in range(1, len(parameters)):
            parameters_avg[key] += parameters[i][key]
        parameters_avg[key] = torch.div(parameters_avg[key], len(parameters))
    return parameters_avg
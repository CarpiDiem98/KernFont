import torch
import torch.nn as nn


def get_loss_criterion(loss_name):
    loss_dict = {
        "MSE": nn.MSELoss(),
        "CrossEntropy": nn.CrossEntropyLoss(),
        "Huber Loss": nn.SmoothL1Loss(),
        "MAE": nn.L1Loss(),
    }

    if loss_name in loss_dict:
        return loss_dict[loss_name]
    else:
        raise ValueError("Criterio di loss non valido: {}".format(loss_name))


def get_optimizer(model, optimizar_name, learning_rate):
    optimizer_dict = {
        "Adam": torch.optim.Adam(model.parameters(), lr=learning_rate),
        "AdamW": torch.optim.AdamW(model.parameters(), lr=learning_rate),
        "SGD": torch.optim.SGD(model.parameters(), lr=learning_rate),
    }

    if optimizar_name in optimizer_dict:
        return optimizer_dict[optimizar_name]
    else:
        raise ValueError("Ottimizzatore non valido: {}".format(optimizar_name))

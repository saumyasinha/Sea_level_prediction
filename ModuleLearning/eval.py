import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


def evaluation_metrics(pred, target, mask):
    diff = target - pred
    diff = diff[mask]
    l2loss = (diff ** 2).mean()
    rmse = sqrt(l2loss)

    mae = np.abs(diff).mean()

    return rmse, mae




# def plot():

import numpy as np


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))
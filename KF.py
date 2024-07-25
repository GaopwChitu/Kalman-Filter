import numpy as np


def data_fusion(x1, sigma1, x2, sigma2):
    """
    一个不加权不迭代的数据融合
    :param x1: 待融合行向量x1
    :param sigma1: x1每个分量的分布的方差组成的行向量
    :param x2: 待融合行向量x2
    :param sigma2: x2每个分量的分布的方差组成的行向量
    :return: 融合结果行向量x
    """
    size_1 = len(x1)
    size_2 = len(x2)
    x = np.array([])
    for i in range(size_1):
        k = np.square(sigma1[i]) / (np.square(sigma1[i]) + np.square(sigma2[i]))
        x = np.append(x, x1[i] + k * (x2[i] - x1[i]))
    return x
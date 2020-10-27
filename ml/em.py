# EM 算法
# 2020/09/23
import math
import random

import numpy as np


def load_data(alpha1, u1, sigma1, alpha2, u2, sigma2, length):
    """初始化数据集
        通过服从高斯分布的随机函数为伪造数据集
        alpha1: 高斯分布1的系数
        u1: 高斯分布1的均值
        sigma1:高斯分布1的方差
        alpha2: 高斯分布2的系数
        u2: 高斯分布2的均值
        sigma2:高斯分布2的方差
        length:
    """
    data1 = np.random.normal(u1, sigma1, int(length * alpha1))
    data2 = np.random.normal(u2, sigma2, int(length * alpha2))

    dataSet = []
    dataSet.extend(data1)
    dataSet.extend(data2)

    random.shuffle(dataSet)

    return dataSet


def calcGauss(dataSetArr, uk, sigmak):
    """计算 高斯分布的值"""
    res = (1 / (math.sqrt(2 * math.pi) * sigmak)) * np.exp(
        -1 * (dataSetArr - uk) * (dataSetArr - uk) / (2 * sigmak ** 2))
    return res


def E_step(dataSetArr, alpha1, u1, sigma1, alpha2, u2, sigma2):
    """计算γ~jk的值, EM算法的E步"""
    gamma1 = alpha1 * calcGauss(dataSetArr, u1, sigma1)
    gamma2 = alpha2 * calcGauss(dataSetArr, u2, sigma2)

    gamma = gamma1 + gamma2

    return gamma1 / gamma, gamma2 / gamma


def M_step(dataSetArr, gamma1, gamma2, u1, u2):
    """EM算法的M步"""

    u1_new = np.dot(gamma1, dataSetArr) / np.sum(gamma1)
    u2_new = np.dot(gamma2, dataSetArr) / np.sum(gamma2)

    sigma1_new = math.sqrt(np.dot(gamma1, (dataSetArr - u1) ** 2) / np.sum(gamma1))
    sigma2_new = math.sqrt(np.dot(gamma2, (dataSetArr - u2) ** 2) / np.sum(gamma2))

    alpha1_new = np.sum(gamma1) / len(gamma1)
    alpha2_new = np.sum(gamma2) / len(gamma2)

    return alpha1_new, u1_new, sigma1_new, alpha2_new, u2_new, sigma2_new


def EM_train(dataSetList, max_iter=800):
    """EM算法迭代训练"""
    dataSetArr = np.array(dataSetList)

    alpha1, u1, sigma1, alpha2, u2, sigma2 = 0.5, 0, 1, 0.5, 1, 1

    epoch = 0
    while epoch < max_iter:
        gamma1, gamma2 = E_step(dataSetArr, alpha1, u1, sigma1, alpha2, u2, sigma2)
        alpha1, u1, sigma1, alpha2, u2, sigma2 = M_step(dataSetArr, gamma1, gamma2, u1, u2)

        epoch += 1

    return alpha1, u1, sigma1, alpha2, u2, sigma2


if __name__ == '__main__':
    import time

    start = time.time()
    alpha1, u1, sigma1, alpha2, u2, sigma2, length = 0.3, -2, 0.5, 0.7, 0.5, 1, 1000
    dataSetList = load_data(alpha1, u1, sigma1, alpha2, u2, sigma2, length)
    print('---------------------------')
    print('the Parameters set is:')
    print(f'alpha1: {alpha1}, u1: {u1}, sigma1: {sigma1}, alpha2: {alpha2}, u2: {u2}, sigma2: {sigma2}')
    alpha1, u1, sigma1, alpha2, u2, sigma2 = EM_train(dataSetList)
    print('the Parameters predict is:')
    print(f'alpha1: {alpha1}, u1: {u1}, sigma1: {sigma1}, alpha2: {alpha2}, u2: {u2}, sigma2: {sigma2}\n')
    end = time.time()
    print(f"elapsed time : {(end-start)}")

# 感知机
# 2020/08/26
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def perceptron(X, y, max_iter=10):
    """
        感知机 算法
    """
    X = np.mat(X)
    y = np.mat(y).T
    n, m = X.shape
    w = np.zeros((1, m))
    b = 0

    h = 0.0001
    for k in range(max_iter):
        for i in range(n):
            xi = X[i]
            yi = y[i]
            if (-1) * yi * (w * xi.T + b) >= 0:
                w = w + h * yi * xi
                b = b + h * yi

        print(f"Epoch {k} training")

    return w, b


def model_test(X, y, w, b):
    X = np.mat(X)
    y = np.mat(y).T
    n, m = X.shape
    errorCnt = 0
    for i in range(n):
        xi = X[i]
        yi = y[i]
        if -1 * yi * (w * xi.T + b) >= 0:
            errorCnt += 1

    return 1 - (errorCnt / n)


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    y_ = []
    for i in y:
        if i == 0:
            y_.append(1)
        else:
            y_.append(-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y_)
    w, b = perceptron(X_train, y_train)
    print(w, b)

    print(f"模型准确率: {model_test(X_test, y_test, w, b)}")

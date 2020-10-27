# 逻辑斯谛回归模型
# 2020/09/04
import numpy as np


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (1 + np.exp(z))


def logistic_regression(trainDataArr, trainLabelArr, max_iter=500):
    # 梯度步长
    h = 0.001

    w = np.zeros(trainDataArr.shape[1])
    # 迭代
    for epoch in range(max_iter):
        for i in range(len(trainDataArr)):
            xi = trainDataArr[i]
            yi = trainLabelArr[i]
            wx = np.dot(xi, w)
            w += h * (yi - sigmoid(wx)) * xi

        if epoch % 100 == 0:
            print(f"epoch = {epoch }\t w = {w}")
    return w


def score(X_test, y_test, w):
    errorCnt = 0
    for i in range(len(X_test)):
        xi = X_test[i]
        y = sigmoid(np.dot(xi, w))
        yi = y_test[i]
        if (y >= 0.5 and yi == 0) or (y < 0.5 and yi == 1):
            errorCnt += 1
    return 1 - errorCnt / len(X_test)


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    # from sklearn.linear_model import LogisticRegression

    X, y = load_breast_cancer(return_X_y=True)
    X = X.tolist()
    for x in X:
        x.append(1)
    X = np.array(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    w = logistic_regression(X_train, y_train)
    print(f"准确率为: {score(X_test, y_test, w)}")


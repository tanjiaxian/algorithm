# 朴素贝叶斯
# 2020/08/31
import numpy as np
import pandas as pd


def naive_bayes(Py, Px_y, x):
    """通过朴素贝叶斯进行概率估计"""
    # 特征数目
    featureNum = 4
    # 类别数目
    classNum = 3
    P = [0] * classNum
    for i in range(classNum):
        sum = 0
        for j in range(featureNum):
            sum += Px_y[i][j][x[j]]
        P[i] = sum + Py[i]
    return P.index(max(P))


def get_all_probability(X, y):
    """通过训练集计算先验概率和条件概率"""
    assert isinstance(X, (np.ndarray, pd.DataFrame))
    n, m = X.shape
    classNum = 3  # iris 一共有三个类别
    # 计算类别先验概率
    Py = np.zeros((classNum, 1))
    for i in range(classNum):
        # print((np.sum(np.mat(y) == i) + 1))
        # print(n+classNum)
        Py[i] = (np.sum(np.mat(y) == i) + 1) / (n + classNum)

    Py = np.log(Py)
    # print(Py)
    # 计算条件概率Px_y = P(X=x | Y=y)
    Px_y = np.zeros((classNum, m, 3))
    for i in range(len(y)):
        label = y[i]
        x = X[i]
        for j in range(m):
            Px_y[label][j][x[j]] += 1
    # print(Px_y)
    for label in range(classNum):
        for j in range(m):
            Px_y0 = Px_y[label][j][0]
            Px_y1 = Px_y[label][j][1]
            Px_y2 = Px_y[label][j][2]

            Px_y[label][j][0] = np.log((Px_y0+1)/(Px_y0+Px_y1+Px_y2+3))
            Px_y[label][j][1] = np.log((Px_y1+1)/(Px_y0+Px_y1+Px_y2+3))
            Px_y[label][j][2] = np.log((Px_y2+1)/(Px_y0+Px_y1+Px_y2+3))

    return Py, Px_y


def model_test(Py, Px_y, X, y):
    """测试集验证"""
    errorCnt = 0
    for i in range(len(X)):
        pred = naive_bayes(Py, Px_y, X[i])
        if pred != y[i]:
            errorCnt += 1
    return 1 - (errorCnt/len(X))


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    # print(X, X.shape)
    # print(y, y.shape)

    t0 = pd.cut(X[:, 0], bins=3, labels=[0, 1, 2])
    t1 = pd.cut(X[:, 1], bins=3, labels=[0, 1, 2])
    t2 = pd.cut(X[:, 2], bins=3, labels=[0, 1, 2])
    t3 = pd.cut(X[:, 3], bins=3, labels=[0, 1, 2])

    t0 = np.transpose(np.array([t0]))
    t1 = np.transpose(np.array([t1]))
    t2 = np.transpose(np.array([t2]))
    t3 = np.transpose(np.array([t3]))

    X = np.hstack([t0, t1, t2, t3])
    # print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    Py, Px_y = get_all_probability(X, y)
    print(f"准确率为: {model_test(Py,Px_y, X_test, y_test)}")
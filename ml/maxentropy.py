# 最大熵模型
# 2020/09/08
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data():
    X, y = load_breast_cancer(return_X_y=True)
    n, m = X.shape
    train = np.zeros((n, m))
    for i in range(m):
        t = pd.cut(X[:, i], bins=3, labels=[0, 1, 2])
        train[:, i] = np.array(t)
    print(train)
    print(train.shape)
    X_train, X_test, y_train, y_test = train_test_split(train, y)
    return (X_train, y_train), (X_test, y_test)


class MaxEntropy:
    """最大熵模型类"""

    def __init__(self, trainDataArr, trainLabelArr, max_iter=100):
        self.trainDataArr = trainDataArr
        self.trainLabelArr = trainLabelArr

        self.featureNum = len(trainDataArr[0])
        self.n = 0  # (x, y)对
        self.N = len(self.trainDataArr)
        self.M = 1000
        self.fixy = self.calc_fixy()  # 计算(x,y)特征对出现的次数
        self.w = [0] * self.n  # Pw(y|x)中的w
        self.xy2idDict, self.id2xyDict = self.createSearchDict()  # (x, y)->id和id->(x, y)的搜索字典
        self.max_iter = max_iter
        self.Ep_xy = self.calcEp_xy()

    def calc_fixy(self):
        """计算特征对(x, y)在训练集中出现的次数"""
        fixyDict = [defaultdict(int) for _ in range(self.featureNum)]

        for i in range(len(self.trainDataArr)):
            for j in range(self.featureNum):
                x = self.trainDataArr[i]
                y = self.trainLabelArr[i]
                fixyDict[j][(x[j], y)] += 1
        for t in fixyDict:
            self.n += len(t)
        return fixyDict

    def createSearchDict(self):
        """
            创建查询字典
            xy2idDict: 通过(x, y)对找到其id,所有出现的xy对都有一个id
            id2xyDict: 通过id找到对应的(x, y)对
        """
        xy2idDict = [{} for _ in range(self.featureNum)]
        id2xyDict = {}
        index = 0
        for feature in range(self.featureNum):
            for (x, y) in self.fixy[feature]:
                xy2idDict[feature][(x, y)] = index
                id2xyDict[index] = (x, y)

                index += 1
        return xy2idDict, id2xyDict

    def calcEpxy(self):
        """特征函数f(x,y)关于模型P(Y|X)与经验分布P~(X)的期望值, 用Ep(f)表示"""
        Epxy = [0.0] * self.n
        for i in range(self.N):
            x = self.trainDataArr[i]
            Pwy_x = [0.0] * 2
            Pwy_x[0] = self.calcPwy_x(x, 0)
            Pwy_x[1] = self.calcPwy_x(x, 1)

            for feature in range(self.featureNum):
                for j in range(2):
                    if (x[feature], j) in self.fixy[feature]:
                        _id = self.xy2idDict[feature][(x[feature], j)]
                        Epxy[_id] += 1 / self.N * Pwy_x[j]
        return Epxy

    def calcEp_xy(self):
        """特征函数f(x, y)关于经验分布P~(x, y)的期望值,用Ep~(f)表示"""
        Ep_xy = [0.0] * self.n
        for feature in range(self.featureNum):
            for (x, y) in self.fixy[feature]:
                _id = self.xy2idDict[feature][(x, y)]

                Ep_xy[_id] = self.fixy[feature][(x, y)] / self.N
        return Ep_xy

    def calcPwy_x(self, X, y):
        """
            计算最大熵模型公式 6.22
            Pw(y|x) = 1 / Zw(x) * np.exp(np.sum(self.w))
        """
        numerator = 0  # 分子
        Z = 0  # 分母

        for i in range(self.featureNum):
            if (X[i], y) in self.xy2idDict[i]:
                _id = self.xy2idDict[i][(X[i], y)]
                numerator += self.w[_id]
            if (X[i], 1 - y) in self.xy2idDict[i]:
                _id = self.xy2idDict[i][(X[i], 1 - y)]
                Z += self.w[_id]

        numerator = np.exp(numerator)
        Z = np.exp(Z) + numerator
        return numerator / Z

    def maxEntropyTrain(self):
        """1.改进的迭代尺度法"""
        import time
        for epoch in range(self.max_iter):
            iterStart = time.time()

            Epxy = self.calcEpxy()
            sigmaList = [0.0] * self.n
            for j in range(self.n):
                sigmaList[j] = (1 / self.M) * np.log(self.Ep_xy[j] / Epxy[j])
            self.w = [w + v for w, v in zip(self.w, sigmaList)]

            iterEnd = time.time()
            # 打印运行时长信息
            print('iter:%d:%d, time:%d' % (epoch, self.max_iter, iterStart - iterEnd))

    def predict(self, X):
        """预测样本X的标记"""
        result = np.zeros(2)
        for j in range(2):
            result[j] = self.calcPwy_x(X, j)
        return np.argmax(result)

    def score(self, testDataArr, testLabelArr):
        """评价模型"""
        errorCnt = 0
        for i in range(len(testDataArr)):
            x = testDataArr[i]
            y = testLabelArr[i]

            if y != self.predict(x):
                errorCnt += 1
        return 1 - errorCnt / len(testDataArr)


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_data()
    clf = MaxEntropy(X_train, y_train, max_iter=250)
    clf.maxEntropyTrain()
    print(f"准确率为: {clf.score(X_test, y_test)}")

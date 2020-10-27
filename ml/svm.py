# 支持向量机
# 2020/09/14
import math
import random

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def load_data():
    """加载数据"""
    X, _y = load_breast_cancer(return_X_y=True)
    y = []
    for i in _y:
        if i == 0:
            y.append(-1)
        else:
            y.append(1)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


class SVM:
    """支持向量机模型"""

    def __init__(self, trainDataArr: np.ndarray, trainLabelArr: np.ndarray, sigma=10, C=200, toler=1e-3):
        """
        SVM模型相关参数
        :param trainDataArr: 训练集
        :param trainLabelArr: 标签集 {1, -1}
        :param sigma: 高斯分母中σ
        :param C: 软间隔惩罚项系数
        :param toler: 精度ε
        """
        self.trainDataArr = np.mat(trainDataArr)
        self.trainLabelArr = np.mat(trainLabelArr).T
        self.sigma = sigma
        self.C = C
        self.toler = toler

        self.m, self.n = np.shape(self.trainDataArr)  # self.m 表示训练集数目, self,n 表示 特征数目
        self.k = self.calcKernel()  # 计算核函数
        self.alpha = [0] * self.m
        self.b = 0
        self.E = [0 * self.trainDataArr[i, 0] for i in range(self.trainLabelArr.shape[0])]  # SMO运算过程中的Ei
        self.supportVecIndex = []

    def calcKernel(self):
        """计算样本集的高斯核"""
        k = [[0 for i in range(self.m)] for j in range(self.m)]
        for i in range(self.m):
            if i % 100 == 0:
                print(f'construct the kernel: {i}:{self.m}')
            X = self.trainDataArr[i, :]
            for j in range(self.m):
                Z = self.trainDataArr[j, :]
                result = (X - Z) * (X - Z).T
                result = np.exp(-1 * result / (2 * self.sigma * self.sigma))
                k[i][j] = result
                k[j][i] = result
        return k

    def isSatisfyKKT(self, i):
        """查看第i个α是否满足KKT条件
            统计学习方法 129页,7.111, 7.112, 7.113公式
            αi == 0 <==> yi * g(xi) >= 1
            0 < αi < C <==> yi * g(xi) == 1
            αi == C <==> yi * g(xi) <= 1
        """
        gxi = self.calc_gxi(i)
        yi = self.trainLabelArr[i]
        alpha = self.alpha[i]

        if math.fabs(alpha) < self.toler and yi * gxi >= 1:
            return True
        elif math.fabs(alpha) > -self.toler and alpha < (self.C + self.toler) and math.fabs(
                yi * gxi - 1) < self.toler:
            return True
        elif math.fabs(alpha - self.C) < self.toler and yi * gxi <= 1:
            return True

        return False

    def calc_gxi(self, i):
        """计算g(xi)
            统计学习方法 127页,7.104公式
            g(x) = Σ αi * yi * K(xi, x) + b
        """
        gxi = 0
        index = [k for k, alpha in enumerate(self.alpha) if alpha != 0]

        for j in index:
            gxi += self.alpha[j] * self.trainLabelArr[j] * self.k[j][i]

        gxi += self.b
        return gxi

    def calc_Ei(self, i):
        """计算 Ei
             统计学习方法 127页,7.105公式
             Ei = g(xi) - yi = {Σ αi * yi * K(xi, x) + b} - yi
        """
        gxi = self.calc_gxi(i)
        return gxi - self.trainLabelArr[i]

    def getAlpahJ(self, E1, i):
        """SMO 选择第二个变量"""
        E2 = 0
        maxE1_E2 = -1
        maxIndex = -1

        nonzeroE = [k for k, Ei in enumerate(self.E) if Ei != 0]
        for j in nonzeroE:
            E2_tmp = self.calc_Ei(j)
            if math.fabs(E1 - E2_tmp) > maxE1_E2:
                maxE1_E2 = math.fabs(E1 - E2_tmp)
                E2 = E2_tmp
                maxIndex = j

        if maxIndex == -1:
            maxIndex = i
            while maxIndex == i:
                maxIndex = int(random.uniform(0, self.m))

            E2 = self.calc_Ei(maxIndex)

        return E2, maxIndex

    def train(self, max_iter=100):
        """依据训练集与标签集训练模型"""
        import time
        epoch = 0
        paramChanged = 1
        while epoch < max_iter and paramChanged > 0:
            epoch += 1
            paramChanged = 0  # 新的一轮将参数改变标志位重新置0
            start = time.time()
            for i in range(self.m):

                if not self.isSatisfyKKT(i):  # 不满足KKT条件

                    E1 = self.calc_Ei(i)
                    E2, j = self.getAlpahJ(E1, i)

                    alpha_old1 = self.alpha[i]
                    alpha_old2 = self.alpha[j]

                    y1 = self.trainLabelArr[i]
                    y2 = self.trainLabelArr[j]

                    if y1 != y2:
                        L = max(0, alpha_old2 - alpha_old1)
                        H = min(self.C, self.C + alpha_old2 - alpha_old1)
                    elif y1 == y2:
                        L = max(0, alpha_old2 + alpha_old1 - self.C)
                        H = min(self.C, alpha_old2 + alpha_old1)

                    if L == H:  # 如果两者相等，说明该变量无法再优化，直接跳到下一次循环
                        continue
                    K11 = self.k[i][i]
                    K22 = self.k[j][j]
                    K12 = self.k[i][j]
                    K21 = self.k[j][i]
                    alpha_new2 = alpha_old2 + y2 * (E1 - E2) / (K11 + K22 - 2 * K12)

                    if alpha_new2 > H:
                        alpha_new2 = H
                    elif alpha_new2 < L:
                        alpha_new2 = L

                    alpha_new1 = alpha_old1 + y1 * y2 * (alpha_old2 - alpha_new2)

                    # 计算阈值b 和差值Ei
                    b1_new = -E1 - y1 * K11 * (alpha_new1 - alpha_old1) - y2 * K21 * (alpha_new2 - alpha_old2) + self.b

                    b2_new = -E2 - y1 * K12 * (alpha_new1 - alpha_old1) - y2 * K22 * (alpha_new2 - alpha_old2) + self.b

                    if (alpha_new1 > 0) and (alpha_new1 < self.C):
                        self.b = b1_new
                    elif (alpha_new2 > 0) and (alpha_new2 < self.C):
                        self.b = b2_new
                    else:
                        self.b = (b1_new + b2_new) / 2

                    self.alpha[i] = alpha_new1
                    self.alpha[j] = alpha_new2

                    self.E[i] = self.calc_Ei(i)
                    self.E[j] = self.calc_Ei(j)

                    if math.fabs(alpha_new2 - alpha_old2) >= self.toler:
                        paramChanged += 1

            end = time.time()
            print(f"epoch = {epoch}: {max_iter} elapsed time = {end-start}s paramChanged = {paramChanged}")

        for i in range(self.m):
            if self.alpha[i] > 0:
                self.supportVecIndex.append(i)

    def calcSingelKernel(self, x1, x2):
        """计算一对高斯核"""
        result = (x1 - x2) * (x1 - x2).T
        result = np.exp(-1 * result / (2 * self.sigma * self.sigma))
        return result

    def predict(self, x):
        """预测样本X的标签"""
        res = 0
        for i in self.supportVecIndex:
            tmp = self.calcSingelKernel(self.trainDataArr[i, :], np.mat(x))
            res += self.alpha[i] * self.trainLabelArr[i] * tmp

        res += self.b
        return np.sign(res)

    def score(self, X_test, y_test):
        """评价测试集"""
        errorCnt = 0
        for i in range(len(X_test)):
            xi = X_test[i]
            yi = y_test[i]

            if self.predict(xi) != yi:
                errorCnt += 1
        return 1 - errorCnt / len(X_test)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    svm = SVM(X_train, y_train)
    svm.train()
    print(f"准确率为: {svm.score(X_test, y_test)}")

    clf = SVC(C=200)
    clf.fit(X_train,y_train)
    print(f"准确率为: {clf.score(X_test, y_test)}")

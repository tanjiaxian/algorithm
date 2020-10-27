# 决策树 ID3 C4.5算法
# 2020/08/31
import numpy as np
import pandas as pd


def calc_H_D(trainLabelArr):
    """计算数据集D的经验熵H(D)"""
    assert isinstance(trainLabelArr, np.ndarray)
    H_D = 0
    # trainLabelArr的类别
    trainLableSet = {label for label in trainLabelArr}

    for i in trainLableSet:
        p = trainLabelArr[trainLabelArr == i].size / trainLabelArr.size
        H_D += -1 * p * np.log2(p)

    return H_D


def calcH_D_A(trainDataArr_DevFeature, trainLabelArr):
    """计算特征A对数据集D的经验条件熵 H(D|A)"""
    H_D_A = 0
    # 计算特征A的离散取值
    trainDataSet = {i for i in trainDataArr_DevFeature}
    DataSize = trainDataArr_DevFeature.size
    for i in trainDataSet:
        H_D_A += (trainDataArr_DevFeature[trainDataArr_DevFeature == i].size / DataSize) * calc_H_D(
            trainLabelArr[trainDataArr_DevFeature == i])

    return H_D_A


def calcBestFeature_byID3(trainDataArr, trainLabelArr):
    """根据ID3算法 获取信息增益最大特征"""
    trainDataArr = np.array(trainDataArr)
    trainLabelArr = np.array(trainLabelArr)

    featureNum = trainDataArr.shape[1]

    MaxG_D_A = -1
    Max_Feature = -1
    H_D = calc_H_D(trainLabelArr)
    for i in range(featureNum):

        trainDataArr_DevFeature = np.array(trainDataArr[:, i].flat)
        # print(trainDataArr_DevFeature)
        G_D_A = H_D - calcH_D_A(trainDataArr_DevFeature, trainLabelArr)

        if G_D_A > MaxG_D_A:
            MaxG_D_A = G_D_A
            Max_Feature = i

    return Max_Feature, MaxG_D_A


def calcBestFeature_byC45(trainDataArr, trainLabelArr):
    """根据C4.5算法 获取信息增益比的最大特征"""
    trainDataArr = np.array(trainDataArr)
    trainLabelArr = np.array(trainLabelArr)

    featureNum = trainDataArr.shape[1]

    MaxGr_D_A = -1
    Max_Feature = -1
    H_D = calc_H_D(trainLabelArr)
    for i in range(featureNum):

        trainDataArr_DevFeature = np.array(trainDataArr[:, i].flat)
        # print(trainDataArr_DevFeature)
        Gr_D_A = (H_D - calcH_D_A(trainDataArr_DevFeature, trainLabelArr)) / H_D

        if Gr_D_A > MaxGr_D_A:
            MaxGr_D_A = Gr_D_A
            Max_Feature = i

    return Max_Feature, MaxGr_D_A


def majorClass(labelArr):
    """获取标签集中最多数的类别"""
    classDict = {}

    for j in labelArr:
        if j not in classDict:
            classDict[j] = 1
        else:
            classDict[j] += 1

    classDict = sorted(classDict.items(), key=lambda x: x[1], reverse=True)
    return classDict[0][0]


def getSubDataArr(trainDataArr, trainLabelArr, A, a):
    """
    更新数据集和标签集
    :param trainDataArr:
    :param trainLabelArr:
    :param A:  要去除的特征索引
    :param a: 特征值为a的保留样本
    :return:
    """
    retDataArr = []
    retLabelArr = []

    for i in range(len(trainDataArr)):
        data = trainDataArr[i]
        if data[A] == a:
            retDataArr.append(data[:A] + data[A + 1:])
            retLabelArr.append(trainLabelArr[i])

    return retDataArr, retLabelArr


def createTree(trainDataList, trainLabelList):
    """构造决策树"""
    Epsilon = 0.01

    trainLabelSet = {i for i in trainLabelList}
    if len(trainLabelSet) == 1:
        return trainLabelList[0]

    if len(trainLabelList) == 0:
        return majorClass(trainLabelList)

    Ag, EpsilonGet = calcBestFeature_byC45(trainDataList, trainLabelList)

    if EpsilonGet < Epsilon:
        return majorClass(trainLabelList)

    treeDict = {Ag: {}}

    AgSet = {L[Ag] for L in trainDataList}  # 特征Ag的取值个数

    for i in AgSet:
        treeDict[Ag][i] = createTree(*getSubDataArr(trainDataList, trainLabelList, Ag, i))

    return treeDict


def predict(testDataList, tree: dict):
    """
    预测样本
    :param testDataArr:
    :param tree:
    :return:
    """
    while True:
        (key, value), = tree.items()
        if type(tree[key]).__name__ == 'dict':

            dataVal = testDataList[key]
            del testDataList[key]

            tree = value[dataVal]

            if type(tree).__name__ == 'int' or type(tree).__name__ == 'int32':
                return tree
        else:
            return value


def model_test(testDataArr, testLabelArr, tree):
    """模型测试"""
    errorCnt = 0
    n = len(testDataArr)
    for i in range(n):
        data = testDataArr[i]

        if testLabelArr[i] != predict(data, tree):
            errorCnt += 1

    return 1 - (errorCnt / n)


if __name__ == '__main__':
    from sklearn.datasets import load_iris, load_boston
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

    # print(calc_H_D(y_train))  # 1.5817111192999054

    # print(calcBestFeature_byID3(X_train, y_train))
    # print(majorClass(y_train))
    print("iris 数据集: ")
    X_train = X_train.tolist()
    X_test = X_test.tolist()
    # y_train = [int(x) for x in y_train]
    #     # y_test = [int(x) for x in y_test]
    tree = createTree(X_train, y_train)
    print(f"准确率为: {model_test(X_test, y_test,tree)}")

    print("波士顿房价数据集: ")
    X, y = load_boston(return_X_y=True)
    print(X, X.shape)
    print(y, y.shape)
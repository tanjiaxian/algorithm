# 决策树 2 CART回归与分类算法
# 2020/09/01
import numpy as np
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def load_data():

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X = X_train.tolist()
    y = y_train.tolist()
    dataMat = []
    for i, j in zip(X, y):
        i.append(j)
        dataMat.append(i)

    testDataMat = []
    X = X_test.tolist()
    y = y_test.tolist()
    for i, j in zip(X, y):
        i.append(j)
        testDataMat.append(i)

    return np.array(dataMat), np.array(testDataMat)


# 切分数据集，对于特征属性feature，以value作为中点，小于value作为数据集1，大于value作为数据集2
def binSplitData(data, feature, value):
    # nonzero，当使用布尔数组直接作为下标对象或者元组下标对象中有布尔数组时，
    # 都相当于用nonzero()将布尔数组转换成一组整数数组，然后使用整数数组进行下标运算。
    # print(data[:, feature] > value)
    mat1 = data[np.nonzero(data[:, feature] > value)[0], :]
    mat2 = data[np.nonzero(data[:, feature] <= value)[0], :]
    return mat1, mat2


# 找到数据切分的最佳位置，遍历所有特征及其可能取值找到使误差最小化的切分阈值
# 生成叶子节点,即计算属于该叶子的所有数据的label的均值（回归树使用总方差）
def regLeaf(data):
    return np.mean(data[:, -1])


# 误差计算函数：总方差
def regErr(data):
    return np.var(data[:, -1]) * np.shape(data)[0]


# 最佳切分查找函数
def chooseBestSplit(data, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 容许的误差下降值
    tolS = ops[0]
    # 切分的最少样本数
    tolN = ops[1]
    # print(data[:, -1].T.tolist())
    # 如果数据的y值都相等，即属于一个label，则说明已经不用再分了，则返回叶子节点并退出
    if len(set(data[:, -1].T.tolist())) == 1:
        return None, leafType(data)
    # 否则，继续分
    m, n = np.shape(data)
    # 原数据集的误差
    s = regErr(data)
    # 最佳误差(先设为极大值),最佳误差对应的特征的index，和对应的使用的切分值
    best_s = np.inf
    best_index = 0
    best_val = 0
    for feat_index in range(n - 1):
        for val in set(data[:, feat_index].T.tolist()):
            # 根据特征feat_index和其对应的划分取值val将数据集分开
            mat1, mat2 = binSplitData(data, feat_index, val)
            # 若某一个数据集大小小于tolN，则停止该轮循环
            if (np.shape(mat1)[0] < tolN) or (np.shape(mat2)[0] < tolN):
                continue
            new_s = errType(mat1) + errType(mat2)
            if new_s < best_s:
                best_s = new_s
                best_index = feat_index
                best_val = val
    # 如果最佳的误差相较于总误差下降的不多，则停止分支，返回叶节点
    if (s - best_s) < tolS:
        return None, leafType(data)
    # 如果划分出来的两个数据集，存在大小小于tolN的，也停止分支，返回叶节点
    mat1, mat2 = binSplitData(data, best_index, best_val)
    if (np.shape(mat1)[0] < tolN) or (np.shape(mat2)[0] < tolN):
        return None, leafType(data)
    # 否则，继续分支，返回最佳的特征和其选取的值
    return best_index, best_val


# 创建回归树
def createTree(data, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 找到最佳的划分特征以及其对应的值
    feat, val = chooseBestSplit(data, leafType, errType, ops)
    # 若达到停止条件，feat为None并返回数值（回归树）或线性方程（模型树）
    if feat is None:
        return val
    # 若未达到停止条件，则根据feat和对应的val将数据集分开，然后左右孩子递归地创建回归树
    # tree 存储了当前根节点划分的特征以及其对应的划分值，另外，左右孩子也作为字典存储
    rgtree = {}
    rgtree['spInd'] = feat
    rgtree['spVal'] = val
    lset, rset = binSplitData(data, feat, val)
    rgtree['left'] = createTree(lset, leafType, errType, ops)
    rgtree['right'] = createTree(rset, leafType, errType, ops)
    return rgtree


# 判断是否为树
def isTree(obj):
    return (type(obj).__name__ == 'dict')


# 递归函数，找到叶节点平均值，塌陷处理
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


# 剪枝
def prune(tree, testData):
    # 如果测试数据为空，则直接对原树进行塌陷处理
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    # 如果当前节点不是叶子节点的父节点，将test数据分支，然后递归地对左子树和右子树剪枝
    if isTree(tree['left']) or isTree(tree['right']):
        lSet, rSet = binSplitData(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 如果当前节点是叶子节点的父节点，即左右子树都为一个数值而非子树，计算剪枝前后，测试数据在这个父节点出的误差
    # 根据误差是否降低来判断是否剪枝（合并左右叶子节点到其父节点，使该父节点成为新的叶子节点）
    if (not isTree(tree['left'])) and (not isTree(tree['right'])):
        lSet, rSet = binSplitData(testData, tree['spInd'], tree['spVal'])
        # 不剪枝
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + sum(np.power(rSet[:, -1] - tree['right'], 2))
        # 剪枝
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree


if __name__ == '__main__':
    dataMat, testDataMat = load_data()
    tree = createTree(dataMat)
    print(tree)
    print(prune(tree, testDataMat))

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    reg = DecisionTreeRegressor()
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))
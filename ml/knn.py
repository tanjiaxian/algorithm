# k近邻法
# 2020/08/27
import copy
import timeit

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class Node:

    def __init__(self, data=None, lc=None, rc=None, depth=None, space=None):
        self.data = data
        self.depth = depth  # 深度
        self.lc = lc
        self.rc = rc
        self.space = space  # 节点创建之前需要划分的超矩形区域
        self.visited = False


class KdTree:
    """构造平衡Kd树"""

    def __init__(self, trainset):
        """假定trainset不为空"""
        assert isinstance(trainset, np.ndarray)

        self.k = trainset.shape[1]
        self.root = Node(depth=0, space=trainset)

        self._generateNode(self.root)

    def _generateNode(self, curNode: Node):

        space = curNode.space
        if space.shape[0] == 1:
            curNode.data = space[0]
            return
        curFeature = curNode.depth % self.k
        curSet = space[space[:, curFeature].argsort()]
        curMid = int(curSet.shape[0] // 2)
        curNode.data = curSet[curMid]

        if curSet[curMid + 1:].shape[0] > 0:
            curNode.rc = Node(depth=curNode.depth + 1, space=curSet[curMid + 1:])
            self._generateNode(curNode.rc)
        if curSet[:curMid].shape[0] > 0:
            curNode.lc = Node(depth=curNode.depth + 1, space=curSet[:curMid])
            self._generateNode(curNode.lc)

    def search(self, x, topk=5):
        """利用kd树找出x的 topk个最近邻"""
        knn_list = []

        def _recursive(node: Node):
            if not node:
                return

            if not node.visited:
                if x[node.depth % self.k] <= node.data[node.depth % self.k]:
                    _recursive(node.lc)

                else:
                    _recursive(node.rc)
            else:
                return
            node.visited = True

            if len(knn_list) < topk:
                dist = calcDist(x, node.data)
                knn_list.append((dist, node.data))

                if x[node.depth % self.k] <= node.data[node.depth % self.k]:
                    _recursive(node.rc)
                else:
                    _recursive(node.lc)
            else:
                edge_dist = abs(x[node.depth % self.k] - node.data[node.depth % self.k])
                max_dist = max(knn_list, key=lambda x: x[0])
                if edge_dist > max_dist[0]:
                    return
                else:
                    dist = calcDist(x, node.data)
                    if dist < max_dist[0]:
                        index = knn_list.index(max_dist)
                        knn_list[index] = (dist, node.data)

                    if x[node.depth % self.k] <= node.data[node.depth % self.k]:
                        _recursive(node.rc)
                    else:
                        _recursive(node.lc)
            return

        root = copy.deepcopy(self.root)
        _recursive(root)
        return knn_list

    def __repr__(self):
        """"""
        queue = []
        result = []
        if not self.root:
            return ""
        queue.append(self.root)
        while queue:
            node = queue.pop(0)
            result.append((node.data, node.depth))

            if node.lc is not None:
                queue.append(node.lc)
            if node.rc is not None:
                queue.append(node.rc)
        return "->".join([str(i) for i in result])


def calcDist(x1, x2):
    """计算样本x1和x2的之间的距离"""
    # print(x1, x2)
    dist = np.sqrt(np.sum(np.square(x1 - x2)))  # 欧氏距离
    # dist = np.sum(x1-x2) # 曼哈顿距离
    return dist


def knn(X_train, y_train, x, topK):
    """
        预测样本x的标记
    :param X_train: 训练集数据集
    :param y_train: 训练集标签集
    :param x: 样本x
    :param K: 选择参考最近邻样本的数量
    :return: 预测样本x的标记
    """
    distList = [0] * len(X_train)
    for i in range(len(X_train)):
        xi = X_train[i]
        distList[i] = calcDist(xi, x)

    topKList = np.argsort(np.array(distList))[:topK]
    labelList = np.zeros(3)  # iris 只有三类
    for index in topKList:
        labelList[int(y_train[index])] += 1
    print(labelList)
    return np.argmax(labelList)


def test(X_train, y_train, X_test, y_test, topK):
    """
        测试正确率
    :param X_test:
    :param y_test:
    :return: 准确率
    """
    X_train = np.mat(X_train)
    y_train = np.mat(y_train).T
    X_test = np.mat(X_test)
    y_test = np.mat(y_test).T

    errorCnt = 0
    for i in range(len(X_test)):
        xi = X_test[i]
        yi = knn(X_train, y_train, xi, topK)
        if yi != y_test[i]:
            errorCnt += 1

    return 1 - (errorCnt / len(X_test))


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)  # 切割训练集和测试集
    print(f"准确率为 = {test(X_train, y_train, X_test, y_test, topK=13)}")

    # sklearn knn
    model = KNeighborsClassifier(n_neighbors=13)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    # 构造kd树
    topk = 2
    # trainset = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    trainset = X_train
    kt = KdTree(np.array(trainset))
    print(kt)  # [7 2]->[5 4]->[9 6]->[2 3]->[4 7]->[8 1]
    print(kt.k)
    print(timeit.timeit(stmt="sorted(kt.search(X_test[0], topk), key=lambda x: x[0])", number=10000, setup="from __main__ import kt, X_test, topk"))
    print(timeit.timeit(
        stmt="sorted([(calcDist(X_test[0], x), np.array(x)) for x in trainset], key=lambda x: x[0], reverse=False)[:topk]",
        number=10000, setup="from __main__ import X_test, topk, trainset,calcDist,np"))


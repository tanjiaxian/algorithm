# 提升方法 AdaBoost算法
# 2020/09/16
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


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


def calc_e_Gx(trainDataArr, trainLabelArr, n, div, rule, D):
    """计算分类错误率"""
    e = 0  # 初始化误分类误差率为0
    x = trainDataArr[:, n]
    y = trainLabelArr
    predict = []

    if rule == 'LisOne':
        L = 1
        H = -1
    else:
        L = -1
        H = 1

    for i in range(trainDataArr.shape[0]):
        if x[i] < div:
            predict.append(L)
            if y[i] != L:
                e += D[i]
        elif x[i] >= div:
            predict.append(H)
            if y[i] != H:
                e += D[i]

    return np.array(predict), e


def createSigleBoostingTree(trainDataArr, trainLabelArr, D):
    """当前训练数据权值为D的情况下,创建单层提升树"""
    # 获取样本数目以及特征数目
    N, m = trainDataArr.shape

    sigleBoostTree = {}
    sigleBoostTree['e'] = 1  # 初始化分类误差率

    for i in range(m):
        for div in set(trainDataArr[:, i].T.tolist()):

            for rule in ['LisOne', 'HisOne']:
                Gx, e = calc_e_Gx(trainDataArr, trainLabelArr, i, div, rule, D)

                if e < sigleBoostTree['e']:
                    sigleBoostTree['e'] = e
                    sigleBoostTree['div'] = div
                    sigleBoostTree['Gx'] = Gx
                    sigleBoostTree['rule'] = rule
                    sigleBoostTree['feature'] = i
    return sigleBoostTree


def createBosstingTree(trainDataArr, trainLabelArr, treeNum=50):
    """
        创建提升树
    """
    # 每增加一层数后，当前最终预测结果列表
    finally_predict = [0] * len(trainLabelArr)
    # 获得训练集数量以及特征个数
    N, m = trainDataArr.shape
    # 初始化训练数据的权值分布
    D = [1 / N] * N

    tree = []
    for i in range(treeNum):
        # 当前权值分布下的计算当前层的提升树
        curTree = createSigleBoostingTree(trainDataArr, trainLabelArr, D)

        alpha = 1/2 * np.log((1-curTree['e'])/curTree['e'])
        Gx = curTree['Gx']

        D = np.multiply(D, np.exp(-1 * alpha * np.multiply(trainLabelArr, Gx))/np.sum(D))
        curTree['alpha'] = alpha
        tree.append(curTree)

        finally_predict += alpha * Gx
        error = sum([1 for i in range(len(trainDataArr)) if np.sign(finally_predict[i]) != trainLabelArr[i]])
        # 计算当前最终误差率
        finally_error = error / N
        if finally_error == 0:
            return tree
        print(f'iter:{i}:{treeNum}, finall error:{finally_error}')

    return tree


def predict(x, div, rule, feature):
    """预测单个样本的标签"""
    if rule == 'LisOne':
        L = 1
        H = -1
    else:
        L = -1
        H = 1

    if x[feature] < div:
        return L
    else:
        return H


def model_test(testDataArr, testLabelArr, tree):
    """模型测试"""
    errorCnt = 0

    for i in range(len(testDataArr)):
        res = 0
        for curTree in tree:
            div = curTree['div']
            rule = curTree['rule']
            feature = curTree['feature']
            alpha = curTree['alpha']

            res += alpha * predict(testDataArr[i], div, rule, feature)

        if np.sign(res) != testLabelArr[i]:
            errorCnt += 1

    return 1 - errorCnt/len(testDataArr)


if __name__ == '__main__':
    from sklearn.ensemble import AdaBoostClassifier

    X_train, X_test, y_train, y_test = load_data()
    tree = createBosstingTree(X_train, y_train)
    print(f"准确率为: {model_test(X_test, y_test, tree)}")

    clf = AdaBoostClassifier(random_state=0)
    clf.fit(X_train, y_train)
    print(f"准确率为: {clf.score(X_test, y_test)}")


# 隐马尔可夫模型
# 2020/09/27
import re
import jieba
import numpy as np


def trainParameter(filename):
    """
    依据训练文本统计 PI, A, B
    :param filename: 训练文本
    :return: 模型参数
    """

    statusDict = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    # 初始化模型参数
    PI = np.zeros(4)
    A = np.zeros((4, 4))
    B = np.zeros((4, 65536))

    with open(filename, encoding='utf8') as f:
        for line in f.readlines():

            line = jieba.cut(line, cut_all=True)
            line = ' '.join(line)

            line = line.strip().split()
            print(line)
            wordLabel = []
            for i in range(len(list(line))):
                if len(line[i]) == 1:
                    Label = 'S'
                else:
                    Label = 'B' + 'M' * (len(line[i]) - 2) + 'E'

                if i == 0:
                    PI[statusDict[Label[0]]] += 1

                for j in range(len(Label)):
                    B[statusDict[Label[j]]][ord(line[i][j])] += 1

                wordLabel.extend(Label)
            # print(wordLabel)
            for i in range(1, len(wordLabel)):
                A[statusDict[wordLabel[i-1]]][statusDict[wordLabel[i]]] += 1

        S = np.sum(PI)
        for i in range(len(PI)):
            if PI[i] == 0:
                PI[i] = -3.14e+100
            else:
                PI[i] = np.log(PI[i]/S)

        for i in range(len(A)):
            S = np.sum(A[i])
            for j in range(len(A[i])):
                if A[i][j] == 0:
                    A[i][j] = -3.14e+100
                else:
                    A[i][j] = np.log(A[i][j]/S)

        for i in range(len(B)):
            S = np.sum(len(B[i]))
            for j in range(len(B[i])):
                if B[i][j] == 0:
                    B[i][j] = -3.14e+100
                else:
                    B[i][j] = np.log(B[i][j]/S)

    return PI, A, B


def loadArticle(filename):
    """
    加载文章
    :param filename: 文件路径
    :return: 文章内容
    """
    artical = []

    with open(filename, encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            # 将该行放入文章列表中
            artical.append(line)
    return artical


def participle(artical, PI, A, B):
    """
    分词
    :param artical: 分词文本
    :param PI: 初始状态概率向量PI
    :param A: 状态转移矩阵
    :param B: 观测概率矩阵
    :return: 分词后的文章
    """
    retActical = []

    for line in artical:
        delta = [[0 for _ in range(4)] for _ in range(len(line))]

        for i in range(4):
            delta[0][i] = PI[i] + B[i][ord(line[0])]

        psi = [[0 for _ in range(4)] for _ in range(len(line))]
        for t in range(1, len(line)):

            for i in range(4):
                tmpDelta = [0] * 4

                for j in range(4):
                    tmpDelta[j] = delta[t-1][j] + A[j][i]

                maxDelta = max(tmpDelta)
                maxDeltaIndex = tmpDelta.index(maxDelta)
                delta[t][i] = maxDelta + B[i][ord(line[t])]
                psi[t][i] = maxDeltaIndex

        sequence = []

        i_opt = delta[len(line)-1].index(max(delta[len(line)-1]))
        sequence.append(i_opt)

        for t in range(len(line)-1, 0, -1):
            i_opt = psi[t][i_opt]
            sequence.append(i_opt)
        sequence.reverse()

        curline = ''
        for i in range(len(line)):
            curline += line[i]

            if (sequence[i] == 3 or sequence[i] == 2) and i != (len(line)-1):
                curline += '|'

        retActical.append(curline)

    return retActical


if __name__ == '__main__':
    PI, A, B = trainParameter('./data/HMMTrainSet.txt')
    # 读取测试文章
    artical = loadArticle('./data/testArtical.txt')

    # 打印原文
    print('-------------------原文----------------------')
    for line in artical:
        print(line)

    # 进行分词
    print('-------------------分词后----------------------')
    ret_artical = participle(artical, PI, A, B)
    for line in ret_artical:
        print(line)
# 图
# 2020/08/07
# author : tanjiaxian
from abc import ABC, abstractmethod
from enum import Enum
from queue import Queue
from typing import Any, List

import numpy as np

from DataStructuresAndAlgorithms.stack import Stack


class VStatus(Enum):
    # 顶点状态
    UNDISCOVERD = 0
    DISCOVERD = 1
    VISITED = 2


class EType(Enum):
    # 边在遍历树中所属的类型

    UNDETERMINED = 0
    TREE = 1
    CROSS = 2
    FORWARD = 3
    BACKWARD = 4


class Vertex:
    """顶点对象"""

    def __init__(self, data: Any):
        self.data = data
        self.inDegree = 0
        self.outDegree = 0
        self.status = VStatus.UNDISCOVERD
        self.dTime = -1
        self.fTime = -1
        self.parent = -1
        self.priority = np.inf

    def __repr__(self):
        return str(self.data)


class Edge:
    """边对象"""

    def __init__(self, data: Any, weight: int):
        self.data = data
        self.weight = weight
        self.type = EType.UNDETERMINED

    def __bool__(self):
        return bool(self.data is not None)

    def __repr__(self):
        return "<- " + str(self.data) + ": " + str(self.weight) + "->"


class PrimPU:
    """针对Prim算法的顶点优先级更新器"""

    def __call__(self, g, uk: int, v: int):  # g: Graph
        if VStatus.UNDISCOVERD == g.status(v):  # 针对uk, 每一尚未被发现的邻接顶点v
            if g.priority(v) > g.weight(uk, v):  # 按Prim策略做松弛
                V = g.V[v]
                V.priority = g.weight(uk, v)  # 更新优先级
                V.parent = uk  # 更新父节点


class DijkstraPU:
    """针对Dijkstra算法的顶点优先级更新器"""

    def __call__(self, g, uk: int, v: int):
        if VStatus.UNDISCOVERD == g.status(v):
            if g.priority(v) > (g.priority(uk) + g.weight(uk, v)):
                V = g.V[v]
                V.priority = g.priority(uk) + g.weight(uk, v)
                V.parent = uk


class Graph(ABC):
    """图Graph模板类"""

    def __init__(self):
        self.n = 0  # 顶点总数
        self.e = 0  # 边总数
        self.V: List[Vertex] = []
        self.E: List[List[Edge]] = []

    def __reset(self):
        """所有顶点,边的辅助信息复位"""
        for i in range(self.n):
            v = self.V[i]
            v.status = VStatus.UNDISCOVERD
            v.dTime = -1
            v.fTime = -1
            v.parent = -1
            v.priority = np.inf
            for j in range(self.n):
                if self.exists(i, j):
                    e = self.E[i][j]
                    e.type = EType.UNDETERMINED

    @abstractmethod
    def insert(self, v):
        """插入顶点, 返回编号"""
        pass

    @abstractmethod
    def remove(self, i: Any):
        """删除顶点及其关联边, 返回该顶点信息"""
        pass

    @abstractmethod
    def vertex(self, i: Any):
        """顶点v的数据(该顶点确实存在)"""
        pass

    @abstractmethod
    def inDegree(self, i: Any):
        """顶点v的入度(该顶点的确存在)"""
        pass

    @abstractmethod
    def outDegree(self, i: Any):
        """顶点v的出度(该顶点的确存在)"""
        pass

    @abstractmethod
    def firstNbr(self, i: Any):
        """顶点v的首个邻接节点"""
        pass

    @abstractmethod
    def nextNbr(self, i: Any, j: Any):
        """顶点v的(相对于顶点u的)下一个邻接顶点"""
        pass

    @abstractmethod
    def status(self, i: Any):
        """顶点v的状态"""
        pass

    @abstractmethod
    def dTime(self, i: Any):
        """顶点v的时间标签dTime"""
        pass

    @abstractmethod
    def fTime(self, i: Any):
        """顶点v的时间标签fTime"""
        pass

    @abstractmethod
    def parent(self, i: Any):
        """顶点v在遍历树中的父亲"""
        pass

    @abstractmethod
    def priority(self, i: Any):
        """顶点v的在遍历树中优先级"""
        pass

    @abstractmethod
    def exists(self, i: Any, j: Any):
        """边(v, u)是否存在"""
        pass

    @abstractmethod
    def insert_edge(self, e, i: Any, j: Any, w):
        """在顶点v和u之间插入权重为w的边e"""
        pass

    @abstractmethod
    def remove_edge(self, i: Any, j: Any):
        """删除顶点v和u之间的边e,并返回该边信息"""
        pass

    @abstractmethod
    def type(self, i: Any, j: Any):
        """边(v, u)的类型"""
        pass

    @abstractmethod
    def edge(self, i: Any, j: Any):
        """边(v, u) 的数据(该边的确存在)"""
        pass

    @abstractmethod
    def weight(self, i: Any, j: Any):
        """边的权重"""
        pass

    def bfs(self, s: int):
        """广度优先搜索算法"""
        assert (0 <= s) and (s < self.n)
        self.__reset()
        self.clock = 0
        v = s
        while True:
            if VStatus.UNDISCOVERD == self.status(v):
                self.__BFS(v)
            v += 1
            v %= self.n
            if s == v:
                break

    def __BFS(self, v: int):
        """(连通域)广度优先搜索算法"""
        Q = Queue()  # 引入辅助队列
        V = self.V[v]
        V.status = VStatus.UNDISCOVERD
        Q.put(v)  # 初始化起点
        while not Q.empty():
            v = Q.get()
            V = self.V[v]
            self.clock += 1
            V.dtime = self.clock  # 取出队首顶点v
            u = self.firstNbr(v)
            while -1 < u:
                U = self.V[u]
                E = self.E[v][u]
                if VStatus.UNDISCOVERD == U.status:  # 若u未被发现,则
                    U.status = VStatus.DISCOVERD  # 发现该顶点
                    Q.put(u)
                    E.type = EType.TREE

                    U.parent = v  # 引入树边,拓展支撑树
                    print(U, end='\t')
                else:  # 若u已被发现,或者甚至已访问完毕
                    E.type = EType.CROSS

                u = self.nextNbr(v, u)

            V.status = VStatus.VISITED

    def dfs(self, s: int):
        """深度优先搜索算法"""
        assert (0 <= s) and (s < self.n)
        self.__reset()
        self.clock = 0
        v = s

        while True:
            if VStatus.UNDISCOVERD == self.status(v):  # 一旦遇到尚未发现的顶点
                self.__DFS(v)

            v += 1
            v %= self.n
            if s == v:
                break

    def __DFS(self, v: int):
        """(连通域)深度优先算法"""
        assert (0 <= v) and (v < self.n)
        V = self.V[v]
        self.clock += 1
        V.dTime = self.clock
        V.status = VStatus.DISCOVERD  # 发现当前顶点v
        u = self.firstNbr(v)
        while -1 < u:
            U = self.V[u]
            E = self.E[v][u]
            if VStatus.UNDISCOVERD == U.status:  # u尚未被发现,意味着支撑树可在此拓展
                E.type = EType.TREE
                U.parent = v
                print(U, end='\t')

                self.__DFS(u)

            elif VStatus.DISCOVERD == U.status:  # u已被发现但尚未访问完毕,应属被后代指向的祖先
                E.type = EType.BACKWARD

            else:  # u已访问完毕(VISITED, 有向图),则视承接关系分为前向边或跨边
                E.type = EType.FORWARD if V.dTime < U.dTime else EType.CROSS

            u = self.nextNbr(v, u)
        V.status = VStatus.VISITED
        self.clock += 1
        V.fTime = self.clock

    def bcc(self, s: Any):
        """基于DFS的双连通分量分解算法"""
        assert (0 <= s) and (s < self.n)
        self.__reset()
        self.clock = 0
        v = s
        S = Stack([], self.n)

        while True:
            if VStatus.UNDISCOVERD == self.status(v):
                self.__BCC(v, S)
                S.pop()

            v += 1
            v %= self.n
            if s == v:
                break

    def __BCC(self, v: int, S: Stack):
        """(连通域)基于DFS的双连通分量分解算法"""
        assert (0 <= v) and (v < self.n)
        self.clock += 1
        V = self.V[v]
        V.dTime = self.clock
        V.fTime = self.clock
        V.status = VStatus.DISCOVERD
        S.push(v)

        u = self.firstNbr(v)
        while -1 < u:
            U = self.V[u]
            E = self.E[v][u]

            if VStatus.UNDISCOVERD == U.status:
                U.parent = v
                E.type = EType.TREE
                print(U, end='\t')
                self.__BCC(u, S)

                if self.fTime(u) < self.dTime(v):
                    V.fTime = min(V.fTime, U.fTime)

                else:

                    while v != S.pop():
                        pass
                    S.push(v)

            elif VStatus.DISCOVERD == U.status:
                E.type = EType.BACKWARD
                if u != V.parent:
                    V.fTime = min(V.fTime, U.dTime)

            else:
                E.type = EType.FORWARD if V.dTime < U.dTime else EType.CROSS

            u = self.nextNbr(v, u)

        V.status = VStatus.VISITED

    def tSort(self, s: int):
        """基于DFS的拓扑排序算法:
            每一个顶点都不会通过边,指向其在此序列中的前驱顶点,这样的一个线性序列,称作原有向图的一个拓扑排序
        """
        assert (0 <= s) and (s < self.n)
        self.__reset()
        self.clock = 0
        v = s
        S = Stack([], self.n)  # 用栈记录排序顶点
        while True:
            if VStatus.UNDISCOVERD == self.status(v):
                if not self.__TSORT(v, S):
                    while not S.empty():  # 任一连通域(亦即整图)非DAG
                        S.pop()
                        break  # 则 不必继续计算,故直接返回
            v += 1
            v %= self.n
            if s == v:
                break

    def __TSORT(self, v: int, S: Stack):
        """(连通域)基于DFS的拓扑排序算法"""
        assert (0 <= v) and (v < self.n)
        V = self.V[v]
        self.clock += 1
        V.dTime = self.clock
        V.status = VStatus.DISCOVERD
        u = self.firstNbr(v)
        while -1 < u:
            U = self.V[u]
            E = self.E[v][u]
            if VStatus.UNDISCOVERD == U.status:
                U.parent = v
                E.type = EType.TREE
                print(U, end='\t')
                if not self.__TSORT(u, S):  # 从顶点u出发深入搜索
                    return False  # 若u及其后代不能拓扑排序(则全图亦必如此),故返回并报告

            elif VStatus.DISCOVERD == U.status:  # 一旦发现后向边(非DAG),则不必深入,故返回报告
                E.type = EType.BACKWARD
                return False

            else:
                E.type = EType.FORWARD if V.dTime > U.dTime else EType.CROSS

            u = self.nextNbr(v, u)

        V.status = VStatus.VISITED
        S.push(self.vertex(v))

        return True  # v及后代可以拓扑排序

    def prim(self, v: Any):
        """最小支撑树"""
        pass

    def dijkstra(self, v: Any):
        """最短路径Dijkstra算法"""
        pass

    def pfs(self, s: int, prioUpdate):
        """优先级搜索框架"""
        assert (0 <= s) and (s < self.n)
        self.__reset()
        v = s
        while True:
            if VStatus.UNDISCOVERD == self.status(v):
                self.__PFS(v, prioUpdate)
            v += 1
            v %= self.n
            if s == v:
                break

    def __PFS(self, s: int, prioUpdate):
        """(连通域)优先级搜素框架"""
        assert (0 <= s) and (s < self.n)
        S = self.V[s]
        S.priority = 0
        S.status = VStatus.VISITED
        S.parent = -1

        while True:
            w = self.firstNbr(s)
            while -1 < w:
                prioUpdate(self, s, w)  # 更新顶点w的优先级及其顶点

                w = self.nextNbr(s, w)

            shortest = np.inf
            w = 0
            while w < self.n:
                if VStatus.UNDISCOVERD == self.status(w):
                    if shortest > self.priority(w):
                        shortest = self.priority(w)
                        s = w
                w += 1
            if VStatus.VISITED == self.status(s):
                break
            S = self.V[s]
            S.status = VStatus.VISITED
            E = self.E[S.parent][s]
            E.type = EType.TREE

            print(S, end='\t')


class GraphMatrix(Graph):

    def __init__(self):
        super(GraphMatrix, self).__init__()
        self.n = 0
        self.e = 0
        self.V: List[Vertex] = []  # 顶点集
        self.E: List[List[Edge]] = []  # 边集

    def vertex(self, i: Any):
        return self.V[i].data

    def inDegree(self, i: Any):
        return self.V[i].inDegree

    def outDegree(self, i: Any):
        return self.V[i].outDegree

    def firstNbr(self, i: Any):
        return self.nextNbr(i, self.n)

    def nextNbr(self, i: Any, j: Any):
        """
            相对于顶点j的下一个邻接顶点(改用邻接表可提高数据)
            逆向线性试探
        """
        while -1 < j:
            j -= 1
            if self.exists(i, j):
                break
        return j

    def status(self, i: Any):
        return self.V[i].status

    def dTime(self, i: Any):
        return self.V[i].dTime

    def fTime(self, i: Any):
        return self.V[i].fTime

    def parent(self, i: Any):
        return self.V[i].parent

    def priority(self, i: Any):
        return self.V[i].priority

    def insert(self, v: Vertex):
        """插入顶点, 返回编号"""
        for i in range(self.n):
            self.E[i].append(None)
        self.n += 1
        self.E.append([None] * self.n)
        self.V.append(v)
        return self.n - 1

    def remove(self, i: Any):
        """删除第i顶点及其关联边(0<=i<n)"""
        for j in range(self.n):  # 所有出边
            if self.exists(i, j):  # 逐条删除
                self.E[i][j] = None
                self.V[j].inDegree -= 1
                self.e -= 1
        self.E.pop(i)  # 删除第i行
        self.n -= 1
        vBak = self.V.pop(i).data
        for j in range(self.n):
            e = self.E[j].pop(i)
            if e:
                e = None
                self.V[j].outDegree -= 1
        return vBak

    def exists(self, i: Any, j: Any):
        return (0 <= i) and (i < self.n) and (0 <= j) and (j < self.n) and (self.E[i][j] is not None)

    def type(self, i: Any, j: Any):
        return self.E[i][j].type

    def edge(self, i: Any, j: Any):
        return self.E[i][j].data

    def weight(self, i: Any, j: Any):
        return self.E[i][j].weight

    def insert_edge(self, e, i: Any, j: Any, w):
        """插入权重为w的边e = (i, j)"""
        if self.exists(i, j):
            return
        self.E[i][j] = Edge(e, w)
        self.e += 1
        self.V[i].outDegree += 1
        self.V[j].inDegree += 1

    def remove_edge(self, i: Any, j: Any):
        """删除顶点i和j之间的连边"""
        if not self.exists(i, j):
            return
        eBak = self.edge(i, j)
        self.E[i][j] = None
        self.e -= 1
        self.V[i].outDegree -= 1
        self.V[j].inDegree -= 1
        return eBak


if __name__ == '__main__':

    graph = GraphMatrix()
    v0 = Vertex(0)
    v1 = Vertex(1)
    v2 = Vertex(2)
    v3 = Vertex(3)
    v4 = Vertex(4)
    v = [v0, v1, v2, v3, v4]
    for x in v: graph.insert(x)

    graph.insert_edge(7, 0, 1, 0.1)
    graph.insert_edge(8, 1, 2, 0.1)
    graph.insert_edge(9, 2, 3, 0.1)
    graph.insert_edge(10, 3, 4, 0.1)
    graph.insert_edge(11, 4, 4, 0.1)

    # graph.remove(0)
    # print(graph.V)
    # print(graph.E)
    # graph.remove_edge(3, 3)
    # print(graph.V)
    # print(graph.E)
    # print(graph.n)
    # print(graph.e)

    graph.bfs(1)
    print()
    graph.dfs(1)
    print()
    graph.tSort(1)
    print()
    graph.bcc(1)
    pu = PrimPU()
    dpu = DijkstraPU()
    print()
    graph.pfs(1, pu)
    print()
    graph.pfs(1, dpu)

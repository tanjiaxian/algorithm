# 图
# 2020/08/07
# author : tanjiaxian
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


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


class Graph(ABC):
    """图Graph模板类"""
    def __init__(self, n: Any, e: Any):
        self.n = n  # 顶点总数
        self.e = e  # 边总数

    def __reset(self):
        """所有顶点,边的辅助信息复位"""
        for i in range(self.n):
            pass

    def __BFS(self):
        """(连通域)广度优先搜索算法"""
        pass

    def _BFS(self):
        """(连通域)深度优先算法"""
        pass

    def __BCC(self):
        """(连通域)基于DFS的双连通分量分解算法"""
        pass

    def __TSORT(self):
        """(连通域)基于DFS的拓扑排序算法"""
        pass

    def __PFS(self, PU):
        """(连通域)优先级搜素框架"""
        pass

    @abstractmethod
    def insert(self, v):
        """插入顶点, 返回编号"""
        pass

    @abstractmethod
    def remove(self, v: Any):
        """删除顶点及其关联边, 返回该顶点信息"""
        pass

    @abstractmethod
    def vertex(self, v: Any):
        """顶点v的数据(该顶点确实存在)"""
        pass

    @abstractmethod
    def inDegree(self, v: Any):
        """顶点v的入度(该顶点的确存在)"""
        pass

    @abstractmethod
    def outDegree(self, v: Any):
        """顶点v的出度(该顶点的确存在)"""
        pass

    @abstractmethod
    def firstNbr(self, v: Any):
        """顶点v的首个邻接节点"""
        pass

    @abstractmethod
    def nextNbr(self, v: Any, u: Any):
        """顶点v的(相对于顶点u的)下一个邻接顶点"""
        pass

    @abstractmethod
    def status(self, v: Any):
        """顶点v的状态"""
        pass

    @abstractmethod
    def dTime(self, v: Any):
        """顶点v的时间标签dTime"""
        pass

    @abstractmethod
    def fTime(self, v: Any):
        """顶点v的时间标签fTime"""
        pass

    @abstractmethod
    def parent(self, v: Any):
        """顶点v在遍历树中的父亲"""
        pass

    @abstractmethod
    def priority(self, v: Any):
        """顶点v的在遍历树中优先级"""
        pass

    @abstractmethod
    def exists(self, v: Any, u: Any):
        """边(v, u)是否存在"""
        pass

    @abstractmethod
    def insert_edge(self, e, v: Any, u: Any, w):
        """在顶点v和u之间插入权重为w的边e"""
        pass

    @abstractmethod
    def remove_edge(self, v: Any, u: Any):
        """删除顶点v和u之间的边e,并返回该边信息"""
        pass

    @abstractmethod
    def type(self, v: Any, u: Any):
        """边(v, u)的类型"""
        pass

    @abstractmethod
    def edge(self, v: Any, u: Any):
        """边(v, u) 的数据(该边的确存在)"""
        pass

    @abstractmethod
    def weight(self, v: Any, u: Any):
        """边的权重"""
        pass

    def bfs(self, v: Any):
        """广度优先搜索算法"""
        pass

    def dfs(self, v: Any):
        """深度优先搜索算法"""
        pass

    def bcc(self, v: Any):
        """基于DFS的双连通分量分解算法"""
        pass

    def tSort(self, v: Any):
        """基于DFS的拓扑排序算法"""
        pass

    def prim(self, v: Any):
        """最小支撑树"""
        pass

    def dijkstra(self, v: Any):
        """最短路径Dijkstra算法"""
        pass

    def pfs(self):
        """优先级搜索框架"""
        pass


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


class Edge:
    """边对象"""
    def __init__(self, data: Any, weight: int):
        self.data = data
        self.weight = weight
        self.type = EType.UNDETERMINED


class GraphMatrix(Graph):

    def __init__(self, n, e):
        super(GraphMatrix, self).__init__(n, e)

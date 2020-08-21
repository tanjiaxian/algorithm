# 优先级队列
# 2020/08/20
from DataStructuresAndAlgorithms.binary_search_tree import Entry
from DataStructuresAndAlgorithms.binarytree import BinTree, BinNode, npl


class PQ:
    """优先级队列PQ模板类"""

    def __init__(self):
        pass

    def insert(self, e: Entry):
        pass

    def getMax(self):
        pass

    def delMax(self):
        pass

    @staticmethod
    def InHeap(n, i):
        """判断PQ[i]是否合法"""
        return (-1 < i) and (i < n)

    @staticmethod
    def Parent(i):
        return (i - 1) // 2

    @staticmethod
    def LastInternal(n):
        """最后一个内部节点(末节点的父亲)"""
        return PQ.Parent(n - 1)

    @staticmethod
    def LChild(i):
        """PQ[i]左孩子"""
        return 1 + (i << 1)

    @staticmethod
    def RChild(i):
        """PQ[i]右孩子"""
        return (1 + i) << 1

    @staticmethod
    def ParentValid(i):
        """判断PQ[i]是否有父亲"""
        return 0 < i

    @staticmethod
    def LChildValid(n, i):
        """判断PQ[i]是否有一个左孩子"""
        return PQ.InHeap(n, PQ.LChild(i))

    @staticmethod
    def RChildValid(n, i):
        """判断PQ[i]是否有两个孩子"""
        return PQ.InHeap(n, PQ.RChild(i))

    @staticmethod
    def Bigger(pq, i, j):
        """取大者"""
        return j if pq[i] < pq[j] else i

    @staticmethod
    def ProperParent(data, n, i):
        """父子(至多)三者中的大者"""
        if PQ.RChildValid(n, i):
            return PQ.Bigger(data, PQ.Bigger(data, i, PQ.LChild(i)), PQ.RChild(i))
        else:
            if PQ.LChildValid(n, i):
                return PQ.Bigger(data, i, PQ.LChild(i))
            else:
                return i


class PQ_ComplHeap(PQ):
    """完全二叉堆"""

    def __init__(self, data):
        super().__init__()
        self._data = data

        self._size = len(self._data)
        self._heapify(self._size)

    def empty(self):
        return 0 >= self._size

    def _percolateDown(self, n, i):
        """对向量前n个词条中的第i个实施下滤"""
        while True:
            j = PQ.ProperParent(self._data, n, i)
            if i != j:
                self._data[i], self._data[j] = self._data[j], self._data[i]
                i = j
            else:
                return i

    def _percolateUp(self, i):
        """上滤"""
        while PQ.ParentValid(i):  # 只要i有父亲(尚未抵达堆顶),则
            j = PQ.Parent(i)  # 将i之父记作j
            if self._data[i] < self._data[j]:
                break
            self._data[i], self._data[j] = self._data[j], self._data[i]
            i = j
        return i

    def _heapify(self, n):
        """Floyd建堆算法"""
        i = PQ.LastInternal(n)
        while PQ.InHeap(n, i):
            self._percolateDown(n, i)
            i -= 1

    def insert(self, e: Entry):
        """将词条插入二叉堆中国"""
        self._data.append(e)
        self._size += 1
        self._percolateUp(self._size - 1)

    def getMax(self):
        """取优先级最高的词条"""
        return self._data[0]

    def delMax(self):
        """删除非空完全二叉堆中优先级最高的词条"""
        if self._size <= 0:
            return
        maxElem = self._data[0]
        self._data[0] = self._data[-1]
        self._data.pop()
        self._size -= 1
        self._percolateDown(self._size, 0)
        return maxElem

    def __repr__(self):
        return "<-".join(["(" + str(i) + ")" for i in self._data])


class PQ_LeftHeap(BinTree, PQ):
    """左式堆"""

    def __init__(self, root: BinNode, data):
        super().__init__(root)
        for e in data:
            self.insert(e)

    def insert(self, e: Entry):
        """基于合并操作的词条插入算法"""
        v = BinNode(e)
        self._root = self._merge(self._root, v)
        self._root.parent = None
        self._size += 1

    def getMax(self):
        return self._root.data

    def delMax(self):
        """基于合并操作的词条删除"""
        lHeap = self._root.lc
        rHeap = self._root.rc

        e = self._root.data
        del self._root
        self._size -= 1
        self._root = self._merge(lHeap, rHeap)
        if self._root:
            self._root.parent = None
        return e

    def _merge(self, a: BinNode, b: BinNode):
        """合并算法: 不失一般性 npl(a)>= npl(b)"""
        if not a:
            return b
        if not b:
            return a
        if npl(a) < npl(b):
            a, b = b, a
        a.npl = npl(a.rc) + 1 if a.rc else 1

        if a.data < b.data:
            a.data, b.data = b.data, a.data  # 一般情况: 首先确保b不大

        a.rc = self._merge(a.rc, b)  # 将a的右子堆,与b合并
        a.rc.parent = a  # 并更新父子关系
        return a  # 返回合并后的堆顶


if __name__ == '__main__':
    pq = PQ_ComplHeap(data=[17, 13, 12, 15, 10, 8, 6, 45, 28, 74])

    print(pq.delMax(), pq.getMax())
    print(pq.delMax(), pq.getMax())
    print(pq.delMax(), pq.getMax())
    print(pq.delMax(), pq.getMax())
    print(pq.delMax(), pq.getMax())
    print(pq.delMax(), pq.getMax())
    print(pq.delMax(), pq.getMax())
    print(pq.delMax(), pq.getMax())
    print(pq.delMax(), pq.getMax())
    print(pq.delMax())
    print(pq)

    lpq = PQ_LeftHeap(root=BinNode(19), data=[17, 13, 12, 15, 10, 8, 6, 45, 28, 74])
    print(lpq.delMax(), lpq.getMax())
    print(lpq.delMax(), lpq.getMax())
    print(lpq.delMax(), lpq.getMax())
    print(lpq.delMax(), lpq.getMax())
    print(lpq.delMax(), lpq.getMax())
    print(lpq.delMax(), lpq.getMax())
    print(lpq.delMax(), lpq.getMax())
    print(lpq.delMax(), lpq.getMax())
    print(lpq)
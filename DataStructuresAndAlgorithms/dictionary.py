# 词典  跳转表
# 2020/8/17
# tanjiaxian
import random
from abc import ABC, abstractmethod
from typing import Any

from DataStructuresAndAlgorithms.binary_search_tree import Entry
from DataStructuresAndAlgorithms.list import List, ListNode


class Dictionary(ABC):

    def __init__(self):
        self._size = 0

    @abstractmethod
    def size(self):
        """当前词条总数"""
        return self._size

    @abstractmethod
    def put(self, k, v):
        """插入词条(禁止雷同词条可能失败)"""
        pass

    @abstractmethod
    def get(self, k):
        """读取词条"""
        pass

    @abstractmethod
    def remove(self, k):
        """删除词条"""
        pass


class QuadlistNode:
    """跳转表模板类"""

    def __init__(self, e=None, pred=None, succ=None, above=None, below=None):
        self.entry = e  #
        self.pred = pred  # 前驱
        self.succ = succ  # 后继
        self.above = above  # 上邻
        self.below = below  # 下邻

    def __bool__(self):
        return bool(id(self) != id(None))

    def __eq__(self, other):
        return self.entry == other.entry

    def __ne__(self, other):
        if not self and not other:
            return True
        elif self and other:
            return self.entry.key != other.entry.key
        else:
            return False

    def insertAsSuccAbove(self, e: Entry, b=None):
        """插入新节点,以当前节点为前驱,以节点b为下邻"""
        x = QuadlistNode(e, self, self.succ, None, b)
        self.succ.pred = x
        self.succ = x
        if b:
            b.above = x
        return x


class Quadlist:

    def __init__(self):
        self._size = 0
        self.header = None
        self.trailer = None

        self.init()  # 如此构造的四联表,不含任何实质的节点,且暂时与其它四联表相互独立

    def init(self):
        """Quanlist初始化,创建Quadlist对象时统一调用"""
        self.header = QuadlistNode()
        self.trailer = QuadlistNode()
        self.header.succ = self.trailer
        self.header.pred = None
        self.trailer.pred = self.header
        self.trailer.succ = None
        self.header.above, self.trailer.above = None, None
        self.header.below, self.trailer.below = None, None
        self._size = 0

    def clear(self):
        """清空Quadlist"""
        oldSize = self._size
        while 0 < self._size:
            self.remove(self.header.succ)
        return oldSize

    def size(self):
        return self._size

    def empty(self):
        return self._size <= 0

    def first(self):
        return self.header.succ  # 首节点位置

    def last(self):
        return self.trailer.pred  # 末节点位置

    def vaild(self, p: QuadlistNode):
        return p and (id(self.trailer) != id(p)) and (id(self.header) != id(p))

    def remove(self, p: QuadlistNode):
        """删除Quadlist内位置p处的节点,返回其中存放的词条"""
        p.pred.succ = p.succ
        p.succ.pred = p.pred
        self._size -= 1
        e = p.entry
        del p
        return e

    def insertAfterAbove(self, e: Any, p: QuadlistNode, b=None):
        self._size += 1
        return p.insertAsSuccAbove(e, b)

    def traverse(self):
        pass


class Skiplist(List, Dictionary):
    """符合Dictionary接口的Skiplist模板类(但隐含假设元素之间可比较大小)"""

    def __init__(self):

        List.__init__(self)

    def skipSearch(self, qlist: ListNode, p: QuadlistNode, k: Any):
        """
            Skiplist词条查找算法
            入口: qlist为顶层列表, p为qlist首节点
            出口:若成功,p为命中关键码所属塔的顶部节点,qlist为p所属列表
            否则,p为所属塔的基座,该塔对应于不大于k的最大且最靠右关键码,qlist不为空
            约定: 多个词条命中时,沿四联表取最靠右者
        """
        while True:

            while p.succ and p.entry.key <= k:
                p = p.succ

            p = p.pred

            if p.pred and k == p.entry.key:
                return True, p, qlist
            qlist = qlist.succ
            if not qlist.succ:
                return False, p, qlist
            p = p.below if p.pred else qlist.data.first()

    def size(self):
        """底层Quadlist规模"""
        return 0 if self.empty() else self.last().data.size()

    def level(self):
        """层高"""
        return self.size()

    def put(self, k, v):
        """插入(注意与Map有别--Skiplist允许词条重复,故必然成功)"""
        e = Entry(k, v)
        if self.empty():
            self.insertAsFirst(Quadlist())
        qlist = self.first()
        p = qlist.data.first()

        flag, p, _ = self.skipSearch(qlist, p, k)
        if flag:
            while p.below:
                p = p.below

        qlist = self.last()

        b = qlist.data.insertAfterAbove(e, p)
        while random.randint(1, 2) % 2:
            while qlist.data.vaild(p) and not p.above:
                p = p.pred
            if not qlist.data.vaild(p):
                if qlist == self.first():
                    self.insertAsFirst(Quadlist())
                p = qlist.pred.data.first().pred

            else:
                p = p.above
            qlist = qlist.pred
            b = qlist.data.insertAfterAbove(e, p, b)
        return True

    def get(self, k):
        """跳转表词条查找算法"""
        if self.empty():
            return
        qlist = self.first()
        p = qlist.data.first()
        flag, p, _ = self.skipSearch(qlist, p, k)
        if not flag:
            return
        return p.entry.value  # 有多个命中时靠后者优先

    def remove(self, k: Any):
        """删除"""
        if self.empty():
            return False

        qlist = self.first()
        p = qlist.data.first()
        flag, p, qlist = self.skipSearch(qlist, p, k)
        if not flag:
            return False

        while True:

            lower = p.below
            qlist.data.remove(p)
            p = lower
            qlist = qlist.succ
            if not qlist.succ:
                break

        while not self.empty() and self.first().data.empty():
            List.remove(self, p=self.first())  # 逐一清除已可能不含词条的顶层Quadlist
        return True

    def __repr__(self):
        qlist = self.last()
        res = []
        while qlist.data:
            p = qlist.data.first()
            while p:
                res.append(p.entry)
                p = p.succ
            qlist = qlist.pred
        # print(res)
        return "->".join([str(i) for i in res])


if __name__ == '__main__':
    # 跳转表测试
    sl = Skiplist()
    print(sl.put(1, 1))
    print(sl.put(2, 1))

    print(sl.put(3, 1))
    print(sl.put(4, 1))
    print(sl.put(5, 1))
    print(sl.put(8, 1))

    print(sl.put(13, 1))
    print(sl.put(21, 1))
    print(sl.put(34, 1))

    print(sl.put(55, 1))
    print(sl.put(89, 1))

    print(sl)
    print(sl.remove(3))
    print(sl.remove(2))
    print(sl)

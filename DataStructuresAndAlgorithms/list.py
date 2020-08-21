# 列表
# 2020/07/31
# author : tanjiaxian
from typing import Any


class ListNode(object):
    """列表节点模板类"""

    def __init__(self, data=None, pred=None, succ=None):
        self.data = data  # 数值
        self.pred = pred  # 前驱
        self.succ = succ  # 后继

    def insertAsSucc(self, value: Any):
        x = ListNode(value, self, self.succ)
        self.succ.pred = x
        self.succ = x  # 设置逆向链接
        return x

    def insertAsPred(self, value: Any):
        x = ListNode(value, self.pred, self)
        self.pred.succ = x
        self.pred = x  # 设置正向链接
        return x

    def __repr__(self):
        return str(self.data)

    def __bool__(self):
        return bool(id(self) != id(None))


class List(object):

    def __init__(self):
        self._header = ListNode()
        self._trailer = ListNode()

        self._header.succ = self._trailer
        self._header.pred = None
        self._trailer.pred = self._header
        self._trailer.succ = None

        self._size = 0

    def first(self):
        """
            首节点位置
        """
        return self._header.succ

    def last(self):
        """
            末节点位置
        """
        return self._trailer.pred

    def empty(self):
        return 0 >= self._size

    @property
    def size(self):
        return self._size

    def valid(self, p: ListNode):
        """判断p的位置是否对外合法"""
        return p and (self._trailer != p) and (self._header != p)

    def __getitem__(self, item):
        assert (item < self._size) and (item >= 0)
        p = self.first()
        while item > 0:
            p = p.succ
            item -= 1
        return p.data

    def find(self, value: Any, n: int, p: ListNode):
        """
            在无序列表内节点P(可能是trailer)的n个(真)前驱中, 找到等于e的最后者
        """
        while n > 0:
            p = p.pred
            if p is None:
                break
            if value == p.data:
                return p
            n -= 1
        return None

    def insertAsFirst(self, value: Any):

        self._size += 1
        return self._header.insertAsSucc(value)  # value当作首节点插入

    def insertAsLast(self, value: Any):
        self._size += 1
        return self._trailer.insertAsPred(value)

    def insertA(self, p: ListNode, value: Any):

        self._size += 1
        return p.insertAsSucc(value)  # value当作p的后继插入

    def insertB(self, p: ListNode, value: Any):

        self._size += 1
        return p.insertAsPred(value)  # value 当作p的前驱插入

    def copyNodes(self, p: ListNode, n: int):
        """
            列表内部方法:复制列表中自位置p起的n项
        """
        x = List()
        while n > 0:
            if p is None:
                break
            x.insertAsLast(p.data)
            p = p.succ
            n -= 1
        return x

    def remove(self, p: ListNode):
        """删除合法节点p,返回其数值"""
        assert p is not None
        data = p.data
        p.pred.succ = p.succ
        p.succ.pred = p.pred
        del p
        self._size -= 1
        return data

    def clear(self):
        """清空列表"""
        oldSize = self._size
        while self._size > 0:
            self.remove(self._header.succ)

        return oldSize

    def deduplicate(self):
        """删除无序列表中的重复节点"""
        if self._size < 2:
            return 0
        oldSize = self._size

        p = self._header.succ
        r = 0

        while self._trailer != p:
            if p is None:
                break
            q = self.find(p.data, r, p)
            if q is None:
                r += 1
            else:
                self.remove(q)
            p = p.succ

        return oldSize - self._size

    def __repr__(self) -> str:
        nums = []
        current = self._header.succ
        while current != self._trailer:
            nums.append(current.data)
            current = current.succ
        return "->".join(str(num) for num in nums)

    def uniquify(self):
        """有序列表--成批剔除重复元素"""
        if self._size < 2:  # 平凡列表自然无重复
            return 0
        oldSize = self._size
        p = self.first()
        q = p.succ

        while self._trailer != q:

            if p.data != q.data:
                p = q
            else:
                self.remove(q)
        return oldSize - self._size

    def search(self, value: Any, n: int, p: ListNode):
        """在有序列表内节点p(可能是trailer)的n个(真)前驱中, 找到不大于e的最后者"""
        assert (n >= 0) and (n < self._size)

        while n >= 0:
            p = p.pred
            if not p.data:  # p.data才是NoneType类型的,可以用not比较
                break
            if p.data <= value:
                break
            n -= 1

        return p

    def insertionSort(self, p: ListNode, n: int):
        """列表的插入排序算法: 对起始于位置p的n个元素排序"""
        # 在任何时刻, 相对于当前节点e=S[r],前缀S[0, r)总是业已有序
        assert (n >= 0) and (n <= self._size)
        for r in range(n):
            self.insertA(self.search(p.data, r, p), p.data)
            p = p.succ
            self.remove(p.pred)

    def selectionSort(self, p: ListNode, n: int):
        """列表的选择排序算法: 对起始于位置p的n个元素排序"""
        # 在任何时刻,后缀S[r, n)已经有序, 且不小于前缀S[0,r)
        assert (n >= 0) and (n <= self._size)

        head = p.pred
        tail = p

        for i in range(n):
            tail = tail.succ

        while 1 < n:

            _max = self.__selectMax(head.succ, n)
            self.insertB(tail, self.remove(_max))
            tail = tail.pred

            n -= 1

    def __selectMax(self, p: ListNode, n: int):
        # 从起始于位置p的n个元素中选出最大者
        _max = p
        cur = p

        while 1 < n:
            cur = cur.succ
            if cur.data > _max.data:
                _max = cur

            n -= 1

        return _max

    def __iter__(self):

        head = self._header

        while head:
            head = head.succ
            yield head.data


if __name__ == '__main__':
    a = List()
    b = List()
    t = a.first()
    for i in range(10):
        a.insertAsFirst(i)
    for i in range(8):
        b.insertAsLast(i)
    # print(a)
    # print(b)

    p = a.first()
    for i in range(3):
        p = p.succ
    # print(p)
    # print(a.empty())
    # print(a.valid(p))
    # a.insertA(p, 1)
    # a.insertB(p, 3)
    # print(a.size)
    # print(a)
    # print(a[3])
    # print(a.copyNodes(p, 5))
    # print(a)
    # print(a.remove(p.pred))
    # print(a.size)
    # print(a.clear())
    # print(a.size)
    # print(a.deduplicate())
    # print(a)
    a.insertionSort(a.first(), 10)
    print(a)
    a.selectionSort(a.first(), 10)
    print(a)
# 数组
# 2020/07/30
# author : tanjiaxian
import random
from typing import Any


class Array(object):

    def __init__(self, capacity: int, data: list):
        self._data = data  # 数据区
        self._capacity = capacity if capacity > len(self._data) else len(self._data)  # 容量

    def copyFrom(self, lo: int, hi: int):
        """
            复制数组区间[lo,hi)
        """
        _elem = []
        _size = 0
        while lo < hi:
            _elem[_size] = self._data[lo]
            _size += 1
            lo += 1
        return _elem

    def __getitem__(self, position: int) -> object:
        return self._data[position]

    def __setitem__(self, index: int, value: object):
        self._data[index] = value

    def __len__(self):
        return len(self._data)

    @property
    def size(self):
        return len(self._data)

    def empty(self):
        if not self._data:
            return True
        return False

    def permute(self):
        """
            置乱算法
        """
        if self.empty():
            return
        d = self._data
        l = len(d)
        for i in range(l):
            _i = random.randint(0, l - 1)
            d[i], d[_i] = d[_i], d[i]

    def unsort(self, lo: int, hi: int):
        """
            区间置乱算法
        """
        if self.empty():
            return
        v = hi - lo
        if v <= 0:
            raise ValueError("hi must be greater than lo")
        d = self._data
        for i in range(lo, hi):
            _i = random.randint(lo, hi - 1)
            d[i], d[_i] = d[_i], d[i]

    def find(self, value: Any, lo: int, hi: int):
        """
            无序查找
        """
        assert (lo >= 0) and (lo < hi) and (hi <= self.size)

        while lo < hi:
            hi -= 1

            if value == self._data[hi]:
                break

        return hi

    def insert(self, index: int, value: Any):
        """
            插入
        """
        _l = len(self)
        if _l >= self._capacity:
            raise ValueError(" The array is full")

        self._data.insert(index, value)

    def remove_section(self, lo: int, hi: int):
        """删除区间[lo,hi)"""
        if lo >= hi:
            return 0
        while hi < len(self._data):
            self._data[lo] = self._data[hi]
            lo += 1
            hi += 1
        self._data = self._data[:lo]  # 截断
        return hi - lo

    def remove(self, r: int):

        e = self._data[r]
        self.remove_section(r, r+1)
        return e

    def pop(self, index: int):

        return self._data.pop(index)

    def deduplicate(self):
        """
            删除无序向量中重复元素
        """
        data = self._data
        oldSize = len(data)
        i = 1
        while i < len(data):
            if self.find(data[i], 0, i) < 0:
                i += 1
            else:
                self.pop(i)
        return oldSize - len(data)

    def disordered(self):
        """
            判断向量是否有序
        """
        n = 0
        for i in range(1, len(self._data)):
            if self._data[i - 1] > self._data[i]:
                n += 1
        return n  # 向量有序当且仅当n = 0

    def unquify(self):
        """
            有序向量重复元素删除法
        """
        i = 0
        j = 1
        while j < len(self._data):
            if self._data[i] != self._data[j]:
                i += 1
                self._data[i] = self._data[j]
            j += 1
        self._data = self._data[:i + 1]
        return j - i

    def search_section(self, value: Any, lo: int, hi: int):
        """
            有序向量的查找
        """
        return self.__binSearch_A(value, lo, hi) if random.randint(0, 1) else self.__binSearch_B(value, lo, hi)

    def search(self, value: Any):
        """
            有序向量的查找
        """
        if not self._data:
            return -1
        lo = 0
        hi = len(self._data)
        return self.__binSearch_C(value, lo, hi)

    def __binSearch_A(self, value: Any, lo: int, hi: int):
        """
            二分查找版本A
            有序向量区间[lo, hi)内查找元素value
        """
        while lo < hi:
            mi = (lo + hi) // 2
            if value < self._data[mi]:
                hi = mi
            elif value > self._data[mi]:
                lo = mi + 1
            else:
                return mi
        return -1

    def __binSearch_B(self, value: Any, lo: int, hi: int):
        """
            二分查找版本B
            有序向量区间[lo, hi)内查找元素value
        """
        while 1 < hi - lo:
            mi = (lo + hi) // 2
            if value < self._data[mi]:
                hi = mi
            else:
                lo = mi
        return lo if value == self._data[lo] else -1

    def __binSearch_C(self, value: Any, lo: int, hi: int):
        """
            二分查找版本C
            有序向量区间[lo, hi)内查找元素value
        """
        while lo < hi:
            mi = (lo+hi)//2
            if value < self._data[mi]:
                hi = mi
            else:
                lo = mi+1
        return lo -1

    def __iter__(self):
        for item in self._data:
            yield item

    def __str__(self):
        return " ".join([str(i) for i in self._data])


if __name__ == '__main__':
    # [1, 4, 7, 5, 3, 9, 6, 8, 2]
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9]
    a = Array(capacity=11, data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9])
    # print(a)
    # a.permute()
    # a.unsort(2, 8)
    # print(a)
    # print(a.find(-1, 0, 8))
    #
    # print(a.deduplicate())
    # print(a)
    # print(a.unquify())
    print(a.search(8, 1, 7))
    print(a)

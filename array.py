# 数组
# 2020/07/30
# author : tanjiaxian
import random
from typing import Any


class Array(object):

    def __init__(self, capacity: int, data: list):
        self._data = data  # 数据区
        self._capacity = capacity  # 容量

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
        while lo <= hi:
            if value != self._data[hi]:
                hi -= 1

        return hi

    def insert(self, index: int, value: Any):
        """
            插入
        """
        _l = len(self)
        if _l >= self._capacity:
            raise ValueError(" The array is full")

        self._data.insert(index, value)

    def delete(self, lo: int, hi: int):

        if lo >= hi:
            return 0
        
        return hi - lo

    def __str__(self):
        return " ".join([str(i) for i in self._data])


if __name__ == '__main__':
    # [1, 4, 7, 5, 3, 9, 6, 8, 2]
    a = Array(capacity=10, data=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(a)
    # a.permute()
    # a.unsort(2, 8)
    # print(a)
    # print(a.find(-1, 0, 8))
    print(a.delete(3, 5))
    print(a)

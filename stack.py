# 栈与队列
# 2020/07/31
# author : tanjiaxian
import copy
from typing import Any


class Stack:

    def __init__(self, data: list, capacity=100):
        self._data = data
        self._capacity = capacity

    def push(self, value: Any):
        if len(self._data) >= self._capacity:
            raise ValueError("Stack is full")
        self._data.append(value)

    def pop(self):
        if self.empty():
            # raise ValueError("stack is empty")
            return
        return self._data.pop(-1)

    def top(self):
        if self.empty():
            # raise ValueError("stack is empty")
            return
        return self._data[-1]

    def empty(self):
        return len(self._data) <= 0

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return "->".join("(" + str(i) + ")" for i in self._data)

    def clear(self):
        self._data.clear()

    def size(self):
        return len(self._data)

    def find(self, value):

        hi = len(self._data)
        while 0 < hi:
            hi -= 1

            if value == self._data[hi]:
                return hi
        return -1


# 栈的典型应用

# 1.逆序输出
def covert_recursion(s: Stack, n: int, base: int):
    digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    if 0 < n:
        s.push(digit[n % base])
        covert_recursion(s, n // base, base)


def covert_iteration(s: Stack, n: int, base: int):
    digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    while 0 < n:
        remainder = n % base
        # print(remainder)
        s.push(digit[remainder])
        n = int(n / base)
        # print(n)


# 八皇后
class Queen(object):

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return (self.x == other.x) or (self.y == other.y) or (self.x + self.y) == (other.x + other.y) or (
                (self.x - self.y) == (other.x - other.y))

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "(" + str(self.x) + ": " + str(self.y) + ")"


def placeQueens(N: int):
    solu = Stack(data=[], capacity=N)
    q = Queen()
    nCheck = 0
    nSolu = 0
    while True:
        if N <= solu.size() or N <= q.y:
            q = solu.pop()
            q.y += 1
        else:
            while (q.y < N) and (0 <= solu.find(q)):
                q.y += 1
                nCheck += 1
            if N > q.y:
                solu.push(q)
                q = copy.copy(q)
                if N <= solu.size():
                    nSolu += 1
                q.x += 1
                q.y = 0
        if N == solu.size():  # 由鸽巢原理可知：在n行n列的棋盘上至多只能放置n个皇后
            print(solu)
        if (0 >= q.x) and (q.y >= N):
            break
    return nCheck, nSolu


# 队列

class Queue(object):

    def __init__(self, data: list, capacity=100):
        self._data = data
        self._capacity = capacity

    def empty(self):
        return len(self._data) <= 0

    def enqueue(self, value: Any):
        if len(self._data) >= self._capacity:
            raise ValueError("Queue is full")
        self._data.append(value)

    def dequeue(self):
        if self.empty():
            return
        return self._data.pop(0)

    def front(self):
        if self.empty():
            return
        return self._data[0]

    def size(self):
        return len(self._data)

    def clear(self):
        self._data.clear()

    def __repr__(self):
        return "<-".join("(" + str(i) + ")" for i in self._data)


if __name__ == '__main__':
    s = Stack(data=[1, 2, 3, 34, 54, 676, 57, 57])
    # covert_recursion(s, 12345, 8)
    # print(s)
    # s.clear()
    # covert_iteration(s, 12345, 8)
    # print(s)

    # print(placeQueens(8))  # 八皇后问题 详解
    print(s)
    q = Queue(data=[1, 2, 3, 3, 4, 54, 67])
    print(q)

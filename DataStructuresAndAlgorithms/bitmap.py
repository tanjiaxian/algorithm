# 位图
# 2020/08/06
# author : tanjiaxian
from builtins import bytearray

from typing import Optional


class Bitmap:
    def __init__(self, num_bits: int):
        self._num_bits = num_bits
        self._bytes = bytearray(num_bits // 8 + 1)  # 一个字节 8个Bit,, 如果有4个字节 则是32

    def setbit(self, k: int) -> None:
        if k > self._num_bits or k < 1: return
        self._bytes[k // 8] |= (1 << k % 8)

    def testbit(self, k: int) -> Optional[bool]:
        if k > self._num_bits or k < 1: return
        return self._bytes[k // 8] & (1 << k % 8) != 0

    def removebit(self, k: int) -> None:
        if k > self._num_bits or k < 1: return
        self._bytes[k // 8] &= ~(1 << (k % 8))


if __name__ == "__main__":
    bitmap = Bitmap(10)
    bitmap.setbit(1)
    bitmap.setbit(3)
    bitmap.setbit(6)
    bitmap.setbit(7)
    bitmap.setbit(8)

    for i in range(1, 10):
        print(bitmap.testbit(i), end=',')

    print()
    bitmap.removebit(1)
    bitmap.removebit(3)
    for i in range(1, 10):
        print(bitmap.testbit(i), end=',')

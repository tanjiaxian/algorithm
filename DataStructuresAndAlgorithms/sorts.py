# 排序 选取中位数
# 2020/08/21
from DataStructuresAndAlgorithms.array import Array


def _majEleCandidate(A: Array):
    maj = None
    c = 0
    for i in range(A.size):
        if 0 == c:
            maj = A[i]
            c = 1
        else:
            if maj == A[i]:
                c += 1
            else:
                c -= 1
    return maj


def _majEleCheck(A, maj):
    occurence = 0

    for i in range(A.size):
        if A[i] == maj:
            occurence += 1
    return 2 * occurence > A.size


def majority(A: Array):
    """众数查找算法"""
    maj = _majEleCandidate(A)
    return _majEleCheck(A, maj)


def triviaMedian(S1: Array, lo1: int, n1: int, S2: Array, lo2: int, n2: int):
    hi1 = lo1 + n1
    hi2 = lo2 + n2

    S = []
    while lo1 < hi1 and lo2 < hi2:
        while lo1 < hi1 and S1[lo1] <= S2[lo2]:
            S.append(S1[lo1])
            lo1 += 1
        if lo1 == hi1:
            break
        while lo2 < hi2 and S2[lo2] <= S1[lo1]:
            S.append(S2[lo2])
            lo2 += 1

    while lo1 < hi1:
        S.append(S1[lo1])
        lo1 += 1

    while lo2 < hi2:
        S.append(S2[lo2])
        lo2 += 1

    return S[(n1 + n2) // 2]


def median(S1: Array, lo1: int, n1: int, S2: Array, lo2: int, n2: int):
    """序列S1[lo1,lo1+n)和S2[lo2,lo2+n)分别有序, n>0, 数据项可能重复"""
    if n1 > n2:
        return median(S2, lo2, n2, S1, lo1, n1)
    if n2 < 6:
        return triviaMedian(S1, lo1, n1, S2, lo2, n2)

    if 2 * n1 < n2:
        return median(S1, lo1, n1, S2, lo2 + (n2 - n1 - 1) // 2, n1 + 2 - (n2 - n1) % 2)

    mi1 = lo1 + n1 // 2
    mi2a = lo2 + (n1 - 1) // 2
    mi2b = lo2 + n2 - 1 - (n1 // 2)
    
    if S1[mi1] > S2[mi2b]:
        return median(S1, lo1, n1 // 2 + 1, S2, mi2a, n2 - (n1 - 1) // 2)

    elif S1[mi1] < S2[mi2a]:
        return median(S1, mi1, (n1 + 1) // 2, S2, lo2, n2 - (n1 // 2))

    else:
        return median(S1, lo1, n1, S2, mi2a, n2 - (n1 - 1) // 2 * 2)


if __name__ == '__main__':
    A = Array(capacity=100, data=[5, 3, 9, 3, 1, 2, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9, 9])
    print(A.size)
    print(majority(A))

    A = Array(capacity=100, data=[1, 2, 3, 4,10,2,12,20,34])
    B = Array(capacity=100, data=[6, 7, 8, 9, 10,12,15])
    n1 = A.size
    n2 = B.size
    print(n1, n2)
    print(median(A, 0, n1, B, 0, n2))

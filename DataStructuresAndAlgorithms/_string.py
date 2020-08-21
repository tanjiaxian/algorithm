# 串
# 2020/08/21


def kmp(P: str, T: str):
    def _buildNext(P):
        m = len(P)
        j = 0
        N = [0] * m
        t = N[0] = -1

        while j < m - 1:
            if 0 > t or P[j] == P[t]:
                j += 1
                t += 1
                N[j] = t if P[j] != P[t] else N[t]
            else:
                t = N[t]
        return N

    next = _buildNext(P)
    print(next)
    n = len(T)
    m = len(P)
    i = 0  # 文本串
    j = 0  # 模式串

    while j < m and i < n:
        if 0 > j or T[i] == P[j]:
            i += 1
            j += 1
        else:
            j = next[j]
    next.clear()
    return i - j


def bm(P: str, T: str):
    def _buildBC(P):
        bc = [-1] * 256
        j = 0
        m = len(P)
        while j < m:
            bc[ord(P[j])] = j
            j += 1
        return bc

    def _buildSS(P):
        m = len(P)
        ss = [0] * m
        ss[m - 1] = m
        lo = m - 1
        hi = m - 1
        j = lo - 1
        while j >= 0:
            if lo < j and ss[m - hi + j - 1] <= j - lo:
                ss[j] = ss[m - hi + j - 1]
            else:
                hi = j
                lo = min(lo, hi)
                while 0 <= lo and P[lo] == P[m - hi + lo - 1]:
                    lo -= 1
                ss[j] = hi - lo
            j -= 1
        return ss

    def _buildGS(P):
        ss = _buildSS(P)
        m = len(P)
        gs = [m] * m
        j = m - 1
        i = 0
        while j < -1:
            if j + 1 == ss[j]:
                gs[i] = m - j - 1
                i += 1
            j -= 1
        for j in range(m - 1):
            gs[m - ss[j] - 1] = m - j - 1
        ss.clear()
        return gs

    bc = _buildBC(P)
    gs = _buildGS(P)

    i = 0
    while len(T) >= i + len(P):
        j = len(P) - 1
        while P[j] == T[i + j]:
            j -= 1
            if 0 > j:
                break
        if 0 > j:
            break
        else:
            i += max(gs[j], j - bc[ord(T[i + j])])
    gs.clear()
    bc.clear()
    return i


if __name__ == '__main__':
    s = "abc abcdab abcdabcdabde"
    pattern = "bcdabd"
    print(kmp(pattern, s), s.find(pattern))
    print(bm(pattern, s), s.find(pattern))

    s = "hello"
    pattern = "ll"
    print(kmp(pattern, s), s.find(pattern))
    print(bm(pattern, s), s.find(pattern))

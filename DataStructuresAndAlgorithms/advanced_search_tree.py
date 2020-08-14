# 高级搜索树: 伸展树 B-树 红黑树 kd树
# 2020/08/12
# tanjiaxian
from typing import Any

from DataStructuresAndAlgorithms.array import Array
from DataStructuresAndAlgorithms.binary_search_tree import BST
from DataStructuresAndAlgorithms.binarytree import BinNode
from DataStructuresAndAlgorithms.binarytree import RBColor, stature
from DataStructuresAndAlgorithms.stack import Queue


class Splay(BST):
    """基于BST实现Splay(伸展树)"""

    def __init__(self, root: BinNode):
        super().__init__(root)

    def _splay(self, v: BinNode):
        """Splay树伸展算法, 从节点v出发逐层伸展"""
        if not v:
            return

        while v.parent and v.parent.parent:

            p = v.parent
            g = p.parent

            gg = g.parent  # 每轮之后v都以原曾祖父为父
            if BinNode.IsLChild(v):
                if BinNode.IsLChild(p):
                    Splay.attachAsLChild(g, p.rc)
                    Splay.attachAsLChild(p, v.rc)

                    Splay.attachAsRChild(p, g)
                    Splay.attachAsRChild(v, p)

                else:
                    Splay.attachAsLChild(p, v.rc)
                    Splay.attachAsRChild(g, v.lc)

                    Splay.attachAsLChild(v, g)
                    Splay.attachAsRChild(v, p)

            elif BinNode.IsRchild(p):
                Splay.attachAsRChild(g, p.lc)
                Splay.attachAsRChild(p, v.rc)

                Splay.attachAsLChild(p, g)
                Splay.attachAsLChild(v, p)

            else:
                Splay.attachAsRChild(p, v.lc)
                Splay.attachAsLChild(g, v.rc)

                Splay.attachAsRChild(v, g)
                Splay.attachAsLChild(v, p)

            if not gg:
                v.parent = None
            else:
                if g == gg.lc:
                    Splay.attachAsLChild(gg, v)
                else:
                    Splay.attachAsRChild(gg, v)
            super()._updateHeight(g)
            super()._updateHeight(p)
            super()._updateHeight(v)

        p = v.parent
        if p:  # 如果p果真非空,则额外再做一次单旋
            if BinNode.IsLChild(v):
                Splay.attachAsLChild(p, v.rc)
                Splay.attachAsRChild(v, p)
            else:
                Splay.attachAsRChild(p, v.lc)
                Splay.attachAsLChild(v, p)

            super()._updateHeight(p)
            super()._updateHeight(v)

        v.parent = None
        return v

    def search(self, e: Any):
        """在伸展树中查找e"""
        self._hot = None
        p = self.searchIn(self.root, e)
        t = p if p else self._hot
        self.root = self._splay(t)
        return self.root

    def insert(self, e: Any):
        """将关键码e插入伸展树中"""
        if not self.root:  # 处理原树为空的退化情况
            self._size += 1
            self.root = BinNode(e, parent=None)
            return self.root

        x = self.search(e)
        if x:
            if e == x.data:
                return self.root

        self._size += 1
        t = self.root
        if self.root.data < e:
            self.root = BinNode(e, parent=None, lc=t, rc=t.rc)
            t.parent = self.root
            if BinNode.HasRChild(t):
                t.rc.parent = self.root
                t.rc = None
        else:
            self.root = BinNode(e, parent=None, lc=t.lc, rc=t)
            t.parent = self.root
            if BinNode.HasLChild(t):
                t.lc.parent = self.root
                t.lc = None

        super()._updateHeightAbove(t)
        return self.root

    def remove(self, e: Any):
        """从伸展树中删除关键码e"""
        if not self.root:
            return False

        x = self.search(e)
        if not x:
            return False

        if e != x.data:
            return False

        w = self.root
        if not BinNode.HasLChild(self.root):
            self.root = self.root.rc
            if self.root:
                self.root.parent = None

        elif not BinNode.HasRChild(self.root):
            self.root = self.root.lc
            if self.root:
                self.root.parent = None
        else:
            lTree = self.root.lc
            lTree.parent = None
            self.root.lc = None
            self.root = self.root.rc
            self.root.parent = None
            self.search(w.data)

            self.root.lc = lTree
            lTree.parent = self.root

        w.data = None
        del w
        self._size -= 1
        if self.root:
            super()._updateHeight(self.root)
        return True

    @staticmethod
    def attachAsLChild(p: BinNode, lc: BinNode):

        p.lc = lc
        if lc:
            lc.parent = p

    @staticmethod
    def attachAsRChild(p: BinNode, rc: BinNode):

        p.rc = rc
        if rc:
            rc.parent = p


class BTNode:

    def __init__(self, e=None, lc=None, rc=None):
        self.parent = None  # 作为根节点初始值
        self.key = Array(capacity=100, data=[])  # 关键码向量
        self.child = Array(capacity=100, data=[])  # 孩子向量(其长度总比key多一)
        if e is not None:
            self.key.insert(0, e)
            self.child.insert(0, lc)
            self.child.insert(1, rc)

            if lc:
                lc.parent = self
            if rc:
                rc.parent = self
        else:
            self.child.insert(0, None)

    def __repr__(self):
        return str(self.key)

    def __bool__(self):
        return any(self.key._data)

    def traLevel(self):
        """子树层次遍历"""
        q = Queue([])
        q.enqueue(self)
        while not q.empty():
            x = q.dequeue()
            print(x.key, end='->')
            for t in x.child:
                if t is None:
                    continue
                q.enqueue(t)


class BTree:
    """B-树模板类"""

    def __init__(self, order=3):
        self._size = 0
        self._order = order
        self._hot = None  # BTree::search最后访问的非空(除非树空)的节点位置
        self._root = BTNode()

    def _solveOverflow(self, v: BTNode):
        """因插入而上溢之后的分裂处理"""
        if self._order >= v.child.size:  # 递归基: 当前节点并未上溢
            return

        s = self._order // 2
        u = BTNode()  # 注意: 新节点已有一个空孩子
        j = 0
        while j < (self._order - s - 1):
            u.child.insert(j, v.child.remove(s + 1))
            u.key.insert(j, v.key.remove(s + 1))
            j += 1
        u.child[self._order - s - 1] = v.child.remove(s + 1)

        if u.child[0] is not None:  # 若u的孩子非空,则令它们的父节点统一
            for j in range(self._order - s):
                u.child[j].parent = u

        p = v.parent
        if not p:
            p = BTNode()
            self._root = p
            p.child[0] = v
            v.parent = p
        r = 1 + p.key.search(v.key[0])
        p.key.insert(r, v.key.remove(s))
        p.child.insert(r + 1, u)
        u.parent = p

        self._solveOverflow(p)

    def _solveUnderflow(self, v: BTNode):
        """因删除而下溢之后的合并处理"""
        if (self._order + 1) // 2 <= v.child.size:
            return

        p = v.parent
        if not p:  # 递归基:已到根节点,没有孩子的下限
            if (not v.key.size) and v.child[0] is not None:
                self._root = v.child[0]
                self._root.parent = None
                v.child[0] = None
                del v
            return

        r = 0
        while p.child[r] != v:
            r += 1

        # 情况一: 向左兄弟借关键码
        if 0 < r:
            ls = p.child[r - 1]
            if (self._order + 1) / 2 < ls.child.size:
                v.key.insert(0, p.key[r - 1])
                p.key[r - 1] = ls.key.remove(ls.key.size - 1)
                v.child.insert(0, ls.child.remove(ls.child.size - 1))

                if v.child[0] is not None:
                    v.child[0].parent = v
                return

        # 情况二: 向右兄弟借关键码
        if (p.child.size - 1) > r:
            rs = p.child[r + 1]
            if (self._order + 1) / 2 < rs.child.size:
                v.key.insert(v.key.size, p.key[r])
                p.key[r] = rs.key.remove(0)
                v.child.insert(v.child.size, rs.child.remove(0))

                if v.child[v.child.size - 1] is not None:
                    v.child[v.child.size - 1].parent = v

                return

        # 情况三: 左右兄弟要么为空(但不可能同时),要么都太"瘦"--合并
        if 0 < r:
            ls = p.child[r - 1]
            ls.key.insert(ls.key.size, p.key.remove(r - 1))
            p.child.remove(r)

            ls.child.insert(ls.child.size, v.child.remove(0))
            if ls.child[ls.child.size - 1] is not None:
                ls.child[ls.child.size - 1].parent = ls

            while not v.key.empty():
                ls.key.insert(ls.key.size, v.key.remove(0))
                ls.child.insert(ls.child.size, v.child.remove(0))

                if ls.child[ls.child.size - 1] is not None:
                    ls.child[ls.child.size - 1].parent = ls

            del v

        else:
            rs = p.child[r + 1]
            rs.key.insert(0, p.key.remove(r))
            p.child.remove(r)

            rs.child.insert(0, v.child.remove(v.child.size - 1))

            if rs.child[0] is not None:
                rs.child[0].parent = rs
            while not v.key.empty():
                rs.key.insert(0, v.key.remove(v.key.size - 1))
                rs.child.insert(0, v.child.remove(v.child.size - 1))
                if rs.child[0] is not None:
                    rs.child[0].parent = rs

            del v

        self._solveUnderflow(p)
        return

    def order(self):
        return self._order

    def size(self):
        return self._size

    def root(self):
        return self._root

    def empty(self):
        return not self._root

    def search(self, e: Any):
        """查找"""
        v = self._root
        self._hot = None
        while v is not None:
            r = v.key.search(e)
            if (0 <= r) and (e == v.key[r]):
                return v
            self._hot = v
            v = v.child[r + 1]
        return

    def insert(self, e: Any):
        """将关键码e插入B树中"""
        v = self.search(e)
        if v is not None:
            return False
        r = self._hot.key.search(e)  # 在节点_hot的有序向量关键码中查找合适的位置插入

        self._hot.key.insert(r + 1, e)
        self._hot.child.insert(r + 2, None)

        self._size += 1
        self._solveOverflow(self._hot)

        return True

    def remove(self, e: Any):
        """从BTree树中删除关键码e"""
        v = self.search(e)
        if not v:
            return False

        r = v.key.search(e)
        if v.child[0] is not None:  # 若u非叶子,则e的后继节点
            u = v.child[r + 1]
            while u.child[0] is not None:
                u = u.child[0]

            v.key[r] = u.key[0]
            v = u
            r = 0

        v.key.remove(r)
        v.child.remove(r + 1)
        self._size -= 1

        self._solveUnderflow(v)
        return True

    def travLevel(self):
        if self._root:
            self._root.traLevel()

    def __repr__(self):
        print("start->", end="")
        # print(self.__dict__)
        self.travLevel()
        print("end", end='\t')
        return f"(***树大小: {self._size}:根节点: {self._root}***)"


class RedBlack(BST):
    """红黑树模板类"""

    def __init__(self, root: BinNode):
        super().__init__(root)

    def _solveDoubleRed(self, x: BinNode):
        """双红修正"""
        if BinNode.IsRoot(x):
            self._root.color = RBColor.RB_BLACK
            self._root.height += 1
            return
        p = x.parent
        if RedBlack.IsBlack(p):
            return
        g = p.parent
        u = BinNode.uncle(x)

        if RedBlack.IsBlack(u):
            if BinNode.IsLChild(x) == BinNode.IsLChild(p):
                p.color = RBColor.RB_BLACK
            else:
                x.color = RBColor.RB_BLACK

            g.color = RBColor.RB_RED

            gg = g.parent

            g = super()._rotateAt(x)  # 旋转后 需要定位 顶部节点 与父节点的关系
            if gg is None:
                self.root = g
            else:
                if gg.data <= g.data:
                    gg.rc = g
                else:
                    gg.lc = g

            g.parent = gg

        else:
            p.color = RBColor.RB_BLACK
            p.height += 1
            u.color = RBColor.RB_BLACK
            u.height += 1

            if not BinNode.IsRoot(g):
                g.color = RBColor.RB_RED

            self._solveDoubleRed(g)

    def _solveDoubleBlack(self, r: BinNode):
        """
            双黑修正: 解决节点x与被其替代的节点均为黑色的问题
        """
        p = r.parent if r else self._hot  # r的父亲
        if not p:
            return
        s = p.rc if r == p.lc else p.lc  # r的兄弟
        if RedBlack.IsBlack(s):
            t = None
            if BinNode.HasLChild(s) and RedBlack.IsRed(s.lc):
                t = s.lc
            elif BinNode.HasRChild(s) and RedBlack.IsRed(s.rc):
                t = s.rc
            if t is not None:
                oldColor = p.color
                p = super()._rotateAt(t)
                if p.parent is None:
                    self.root = p
                else:
                    if p.parent.data <= p.data:
                        p.parent.rc = p
                    else:
                        p.parent.lc = p

                b = p
                if BinNode.HasLChild(b):
                    b.lc.color = RBColor.RB_BLACK
                    self._updateHeight(b.lc)
                if BinNode.HasRChild(b):
                    b.rc.color = RBColor.RB_BLACK
                    self._updateHeight(b.rc)

                b.color = oldColor
                self._updateHeight(b)
            else:
                s.color = RBColor.RB_RED
                s.height -= 1
                if RedBlack.IsRed(p):
                    p.color = RBColor.RB_BLACK
                else:
                    p.height -= 1
                    self._solveDoubleBlack(p)

        else:
            s.color = RBColor.RB_BLACK
            p.color = RBColor.RB_RED
            t = s.lc if BinNode.IsLChild(s) else s.rc

            self._hot = p
            p = super()._rotateAt(t)
            if p.parent is None:
                self.root = p
            else:
                if p.parent.data <= p.data:
                    p.parent.rc = p
                else:
                    p.parent.lc = p

            self._solveDoubleBlack(r)

    def _updateHeight(self, x: BinNode):
        """更新节点高度"""
        x.height = max(stature(x.lc), stature(x.rc))
        if RedBlack.IsBlack(x):
            x.height += 1
        return x.height

    def insert(self, e: Any):
        """插入"""
        x = self.search(e)
        if x:
            return x

        x = BinNode(e, parent=self._hot, lc=None, rc=None)  # 创立红节点x:以self._hot为父节点,黑高度为-1
        x.height = -1

        if not self._hot:
            self._root = x

        else:

            if e < self._hot.data:
                self._hot.lc = x
            else:
                self._hot.rc = x

        self._size += 1
        self._solveDoubleRed(x)

        if x:
            return x
        else:
            return self._hot.parent

    def remove(self, e: Any):
        """删除"""
        x = self.search(e)
        if not x:
            return False
        r = self.removeAt(x)
        self._size -= 1
        if not self._size:
            return True

        if not self._hot:
            self._root.color = RBColor.RB_BLACK
            self._updateHeight(self._root)
            return True
        if RedBlack.BlackHeightUpdated(self._hot):
            return True
        if RedBlack.IsRed(r):
            r.color = RBColor.RB_BLACK
            r.height += 1
            return True
        self._solveDoubleBlack(r)
        return True

    @staticmethod
    def IsBlack(p):
        return (not p) or RBColor.RB_BLACK == p.color  # 外部节点也视为黑节点

    @staticmethod
    def IsRed(p):
        return not RedBlack.IsBlack(p)  # 非黑即红

    @staticmethod
    def BlackHeightUpdated(x):
        """RedBlack高度更新条件"""
        t1 = stature(x.lc) == stature(x.rc)
        if RedBlack.IsRed(x):
            t = stature(x.lc)
        else:
            t = stature(x.lc) + 1
        t2 = (x.height == t)
        return t1 and t2


if __name__ == '__main__':
    # 1. 伸展树 例子
    root = BinNode(36, parent=None)
    # T = Splay(root=None)
    # T.insert(27)
    # T.insert(6)
    # T.insert(58)
    # T.insert(53)
    # T.insert(64)
    # T.insert(40)
    # T.insert(46)
    # print(T)
    # T.insert(39)
    # print(T)
    # T.remove(39)
    # T.remove(46)
    # print(T)

    # x = BTNode(23)
    # y = BTNode(36, lc=x)
    # B-树 例子
    # T = BTree()  # 该实例与数据结构(C++语言版)-邓俊辉 221页 8.15图相同
    # T.insert(53)
    # T.insert(36)
    # T.insert(77)
    # T.insert(89)
    # T.insert(19)
    # T.insert(41)
    # T.insert(51)
    # T.insert(75)
    # T.insert(79)
    # T.insert(97)
    # T.insert(84)
    #
    # T.insert(23)
    # T.insert(29)
    # T.insert(45)
    # T.insert(87)
    # print(T)
    #
    # T.insert(64)
    # print(T)
    # T.remove(41)
    # print(T)
    # T.remove(53)
    # print(T)
    # T.remove(75)
    # print(T)
    # T.remove(84)
    # print(T)
    # T.remove(51)
    # print(T)

    # 红黑树
    T = RedBlack(root=None)
    T.insert(53)
    T.insert(36)
    T.insert(77)
    T.insert(89)
    T.insert(19)
    T.insert(41)
    T.insert(16)
    T.insert(75)
    T.insert(79)
    T.insert(97)
    T.insert(78)
    T.insert(84)
    T.insert(23)
    T.insert(29)
    T.insert(45)
    T.insert(87)
    print(T)
    T.remove(87)
    print(T)
    T.remove(53)
    print(T)
    T.remove(77)
    print(T)
    T.remove(78)
    print(T)
    T.remove(51)
    print(T)

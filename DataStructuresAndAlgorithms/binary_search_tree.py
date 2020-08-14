# 二叉搜索树以及平衡二叉搜索树
# 2020/08/11
# tanjiaxian

from typing import Any

from DataStructuresAndAlgorithms.binarytree import BinTree, BinNode, stature


class Entry(object):
    """词条模板类"""

    def __init__(self, key: Any, value: Any):
        self.key = key
        self.value = value

    def __lt__(self, other):
        return self.key < other.key

    def __gt__(self, other):
        return self.key > other.key

    def __eq__(self, other):
        return self.key == other.key

    def __ne__(self, other):
        return self.key != other.key


class BST(BinTree):
    """由BinTree派生BST模板类"""

    def __init__(self, root: BinNode):
        self._hot = None
        super(BST, self).__init__(root)

    def _connect34(self, a: BinNode, b: BinNode, c: BinNode,
                   T0: BinNode, T1: BinNode, T2: BinNode, T3: BinNode):
        """按照3+4连接3个节点和4个子树,返回重组之后局部子树根节点位置"""
        a.lc = T0
        if T0:
            T0.parent = a
        a.rc = T1
        if T1:
            T1.parent = a
        super(BST, self)._updateHeight(a)
        c.lc = T2
        if T2:
            T2.parent = c
        c.rc = T3
        if T3:
            T3.parent = c
        super(BST, self)._updateHeight(c)
        b.lc = a
        a.parent = b
        b.rc = c
        c.parent = b
        super(BST, self)._updateHeight(b)
        return b

    def _rotateAt(self, v: BinNode):
        """v为非空孙辈节点"""
        p = v.parent
        g = p.parent  # 视v, p, g相对位置分四种情况

        if BinNode.IsLChild(p):
            if BinNode.IsLChild(v):
                p.parent = g.parent  # 向上联接

                return self._connect34(v, p, g, v.lc, v.rc, p.rc, g.rc)
            else:
                v.parent = g.parent
                return self._connect34(p, v, g, p.lc, v.lc, v.rc, g.rc)
        else:
            if BinNode.IsRchild(v):
                p.parent = g.parent
                return self._connect34(g, p, v, g.lc, p.lc, v.lc, v.rc)
            else:
                v.parent = g.parent
                return self._connect34(g, v, p, g.lc, v.lc, v.rc, p.rc)

    def searchIn(self, v: BinNode, e: Any):
        """在以v为根的BST子树中查找关键码e"""
        if not v or e == v.data:  # 递归基: 在节点v处命中
            return v
        self._hot = v
        t = v.lc if e < v.data else v.rc
        return self.searchIn(t, e)

    def search(self, e: Any):
        """在BST中查找"""
        self._hot = None
        return self.searchIn(self._root, e)

    def insert(self, e: Any):
        """将关键码e插入BST树中"""
        x = self.search(e)
        if x:
            return x
        if e <= self._hot.data:
            x = self._hot.insertAsLC(e)
        else:
            x = self._hot.insertAsRC(e)
        self._size += 1
        super(BST, self)._updateHeightAbove(x)
        return x

    def remove(self, e: Any):
        """从BST树中删除关键码e"""
        x = self.search(e)
        if not x:
            return False
        self.removeAt(x)
        self._size -= 1
        super(BST, self)._updateHeightAbove(self._hot)
        return True

    def removeAt(self, x: BinNode):

        w = x  # 实际被摘除的节点,初值同x
        succ = None  # 实际被删除节点的接替者
        if not BinNode.HasLChild(x):  # 若x的左子树为空,则可直接将x替换为其右子树
            succ = x = x.rc

        elif not BinNode.HasRChild(x):  # 若x的右子树为空,对称处理 此时succ != None
            succ = x = x.lc
        else:
            w = w.succ()
            x.data, w.data = w.data, x.data
            u = w.parent
            succ = w.rc
            if u == x:
                u.rc = succ
            else:
                u.lc = succ
        self._hot = w.parent

        if succ:
            succ.parent = self._hot
            if self._hot.data <= succ.data:
                self._hot.rc = succ
            else:
                self._hot.lc = succ
        w.data = None
        del w
        return succ


class AVL(BST):
    """基于BST实现AVL树"""

    def __init__(self, root: BinNode):
        super(AVL, self).__init__(root)

    def insert(self, e: Any):
        """将关键码e插入AVL树中"""
        x = self.search(e)
        if x:
            return x
        if e <= self._hot.data:
            xx = self._hot.insertAsLC(e)
        else:
            xx = self._hot.insertAsRC(e)
        self._size += 1

        g = self._hot
        while g:
            if not AVL.AvlBalanced(g):
                g = self._rotateAt(AVL.tallerChild(AVL.tallerChild(g)))

                if g.parent is None:
                    self.root = g
                else:
                    if g.parent.data <= g.data:
                        g.parent.rc = g
                    else:
                        g.parent.lc = g

                break
            else:
                super()._updateHeight(g)

            g = g.parent
        return xx

    def remove(self, e: Any):
        """从AVL树中删除关键码e"""
        x = self.search(e)
        if not x:
            return False
        self.removeAt(x)
        self._size -= 1
        g = self._hot
        while g:
            if not AVL.AvlBalanced(g):
                # g = BinNode.FromParentTo(g)
                g = self._rotateAt(AVL.tallerChild(AVL.tallerChild(g)))
                if g.parent is None:
                    self.root = g
                else:
                    if g.parent.data <= g.data:
                        g.parent.rc = g
                    else:
                        g.parent.lc = g
            super()._updateHeight(g)
            g = g.parent
        return True

    @staticmethod
    def Balanced(x):
        return stature(x.lc) == stature(x.rc)  # 理想平衡条件

    @staticmethod
    def BalFac(x):
        return stature(x.lc) - stature(x.rc)  # 平衡因子

    @staticmethod
    def AvlBalanced(x):
        return (-2 < AVL.BalFac(x)) and (AVL.BalFac(x) < 2)

    @staticmethod
    def tallerChild(x):
        """在左,右孩子中取更高者"""
        if stature(x.lc) > stature(x.rc):
            return x.lc
        elif stature(x.lc) < stature(x.rc):
            return x.rc
        else:
            if BinNode.IsLChild(x):
                return x.lc
            else:
                return x.rc


if __name__ == '__main__':
    x = BinNode(data=36, parent=None)
    # T = BST(x)
    # T.insert(27)
    # T.insert(6)
    # T.insert(58)
    # T.insert(53)
    # T.insert(64)
    # T.insert(40)
    # T.insert(46)
    # T.insert(39)
    # print(T)
    # print(AVL.AvlBalanced(x))

    # T = AVL(x)
    # T.insert(27)
    # print(AVL.AvlBalanced(T.root), T.root)
    # T.insert(6)
    # print(AVL.AvlBalanced(T.root), T.root)
    # T.insert(58)
    # print(AVL.AvlBalanced(T.root), T.root)
    # T.insert(53)
    # print(AVL.AvlBalanced(T.root), T.root)
    # T.insert(64)
    # print(AVL.AvlBalanced(T.root), T.root)
    # T.insert(40)
    # print(AVL.AvlBalanced(T.root), T.root)
    # T.insert(46)
    # print(T, AVL.AvlBalanced(T.root), T.root)
    # T.insert(39)
    # print(AVL.AvlBalanced(T.root), T.root)
    # print(T, T.root)
    # T.remove(39)
    # print(AVL.AvlBalanced(T.root), T.root)
    # T.remove(40)
    # print(AVL.AvlBalanced(T.root), T.root)
    # T.remove(64)
    # print(AVL.AvlBalanced(T.root), T.root)
    # T.remove(53)
    # print(AVL.AvlBalanced(T.root), T.root)
    # T.remove(58)
    # print(AVL.AvlBalanced(T.root), T.root)
    # T.remove(6)
    # print(AVL.AvlBalanced(T.root), T.root)
    # T.remove(27)
    # print(AVL.AvlBalanced(T.root), T.root)
    # print(T, AVL.AvlBalanced(T.root), T.root)
    # print(T.root)

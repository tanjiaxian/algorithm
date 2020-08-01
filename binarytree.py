# 二叉树
# 2020/08/01
# author : tanjiaxian
from typing import Any
from enum import Enum


def stature(p):
    if not p:
        return -1
    return p.height


class RBColor(Enum):
    RB_RED = "红色"
    RB_BLACK = "黑色"


class BinNode(object):

    def __init__(self, data=None, parent=None, lc=None, rc=None):
        self._data = data
        self._parent = parent  # 父节点
        self._lc = lc  # 左孩子
        self._rc = rc  # 右孩子
        self._height = 0  # 高度
        self._npl = 1  # Null Path Length
        self._color = RBColor.RB_RED  # 颜色(红黑树)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value: int):
        self._height = value

    @property
    def parent(self):
        return self._parent

    @property
    def data(self):
        return self._data

    @property
    def lc(self):
        return self._lc

    @property
    def rc(self):
        return self._rc

    @property
    def npl(self):
        return self._npl

    @property
    def color(self):
        return self._color

    def size(self):
        """统计当前节点后代总数，亦即以其为根的子树的规模"""
        pass

    def insertAsLC(self, value:Any):
        """作为当前节点的左孩子插入新节点"""
        lc = BinNode(value, self)
        return lc

    def insertAsRC(self,value:Any):
        """作为当前节点的右孩子插入新节点"""
        rc = BinNode(value, self)
        return rc

    def succ(self):
        """取当前节点的直接后继"""
        pass

    def traLevel(self):
        """子树层次遍历"""
        pass

    def traPre(self):
        """子树先序遍历"""
        pass

    def traIn(self):
        """子树中序遍历"""
        pass

    def traPost(self):
        """子树后序遍历"""
        pass

    def __eq__(self, other):
        return self._data == other.data

    def __ne__(self, other):
        return not(self._data == other.data)

    def __lt__(self, other):
        return self._data < other.data

    def __le__(self, other):
        return self._data <= other.data

    def __gt__(self, other):
        return self._data > other.data

    def __ge__(self, other):
        return self._data >= other.data

    def __repr__(self):
        return str(self._data)

    def __bool__(self):
        return bool(self.data is not None)

    @staticmethod
    def IsRoot(x):
        return not x.parent

    @staticmethod
    def IsLChild(x):
        return (not BinNode.IsRoot(x)) and (x == x.parent.lc)

    @staticmethod
    def IsRchild(x):
        return (not BinNode.IsRoot(x)) and (x == x.parent.rc)

    @staticmethod
    def HasParent(x):
        return not BinNode.IsRoot(x)

    @staticmethod
    def HasLChild(x):
        return x.lc

    @staticmethod
    def HasRChild(x):
        return x.rc

    @staticmethod
    def HasChild(x):
        return BinNode.HasLChild(x) or BinNode.HasRChild(x)

    @staticmethod
    def HasBothChild(x):
        return BinNode.HasLChild(x) and BinNode.HasRChild(x)

    @staticmethod
    def IsLeaf(x):
        return not BinNode.HasChild(x)

    @staticmethod
    def sibling(p):
        # 兄弟
        if BinNode.IsLChild(p):
            return p.parent.rc
        else:
            return p.parent.lc

    @staticmethod
    def uncle(x):
        # 叔叔
        if BinNode.IsLChild(x.parent):
            return x.parent.parent.rc
        else:
            return x.parent.parent.lc

    @staticmethod
    def FromParentTo(x):
        # 来自父亲的引用
        if BinNode.IsRoot(x):
            return x
        else:
            if BinNode.IsLChild(x):
                return x.parent.lc
            else:
                return x.parent.rc


class BinTree(object):

    def __init__(self, root: BinNode):

        self._size = 0
        self._root = root

    def __updateHeight(self, x: BinNode):
        x.height = 1 + max([stature(x.lc), stature(x.rc)])
        return x.height

    def __updateHeightAbove(self, x: BinNode):
        while x:
            self.__updateHeight(x)
            x = x.parent

    def size(self):
        return self._size

    def empty(self):
        return not self._root

    def root(self):
        return self._root

    def insertAsRoot(self, value: Any):
        self._size = 1
        self._root = BinNode(value)


if __name__ == '__main__':

    x = BinNode(data=None, parent=None)
    print(not x)
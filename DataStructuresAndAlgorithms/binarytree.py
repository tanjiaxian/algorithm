# 二叉树
# 2020/08/01
# author : tanjiaxian
from copy import copy
from enum import Enum
from typing import Any

from DataStructuresAndAlgorithms.stack import Stack, Queue


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
        self.height = 0  # 高度
        self.npl = 1  # Null Path Length
        self.color = RBColor.RB_RED  # 颜色(红黑树)


    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, x):
        self._parent = x

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: Any):
        self._data = value

    @property
    def lc(self):
        return self._lc

    @lc.setter
    def lc(self, x):
        self._lc = x

    @property
    def rc(self):
        return self._rc

    @rc.setter
    def rc(self, x):
        self._rc = x

    def size(self):
        """统计当前节点后代总数，亦即以其为根的子树的规模"""
        n = 0
        if not self:
            return 0
        else:
            n += 1
            if self._lc:
                n += self._lc.size()
            if self._rc:
                n += self._rc.size()
            return n

    def insertAsLC(self, value: Any):
        """作为当前节点的左孩子插入新节点"""
        self._lc = BinNode(value, self)
        return self._lc

    def insertAsRC(self, value: Any):
        """作为当前节点的右孩子插入新节点"""
        self._rc = BinNode(value, self)
        return self._rc

    def succ(self):
        """取当前节点的直接后继"""
        if not self:
            return
        s = copy(self)

        if self._rc:
            s = self._rc
            while BinNode.HasLChild(s):
                s = s.lc
        else:
            while BinNode.IsRchild(s):
                s = s.parent
            s = s.parent

        return s

    def traLevel(self):
        """子树层次遍历"""
        q = Queue([])
        q.enqueue(self)
        while not q.empty():
            x = q.dequeue()
            print(x.data, end='->')
            if BinNode.HasLChild(x):
                q.enqueue(x.lc)
            if BinNode.HasRChild(x):
                q.enqueue(x.rc)

    def travPre_R(self):
        """先序遍历"""
        if not self:
            return

        print(self._data, end="->")
        if self._lc:
            self._lc.travPre_R()
        if self._rc:
            self._rc.travPre_R()

    def travPost_R(self):
        """后序遍历"""
        if not self:
            return
        if self._lc:
            self._lc.travPost_R()
        if self._rc:
            self._rc.travPost_R()
        print(self._data, end="->")

    def travIn_R(self):
        """中序遍历"""
        if not self:
            return

        if self._lc:
            self._lc.travIn_R()
        print(self._data, end="->")
        if self._rc:
            self._rc.travIn_R()

    def visitAlongLeftBranch(self, s: Stack):
        """从当前节点出发,沿左分支不断深入,直至没有左分支的节点;沿途节点遇到后立即访问"""
        x = copy(self)
        while x:
            print(x.data, end="->")
            if x._rc:
                s.push(x._rc)
            x = x._lc

    def travPre_I2(self):
        """二叉树先序遍历迭代版本"""
        s = Stack([])
        x = copy(self)
        while True:
            x.visitAlongLeftBranch(s)
            if s.empty():
                break
            x = s.pop()

    def goAlongLeftBranch(self, s: Stack):
        """从当前节点出发,沿分支不断深入,直至没有左分支的节点"""
        x = copy(self)
        while x:
            s.push(x)
            x = x.lc

    def travIn_I1(self):
        """二叉树中序遍历迭代版本1"""
        x = copy(self)
        s = Stack([])
        while True:
            if x:
                x.goAlongLeftBranch(s)
            if s.empty():
                break
            x = s.pop()
            print(x.data, end="->")
            x = x.rc

    def travIn_I2(self):
        """二叉树中序遍历迭代版本2"""
        x = copy(self)
        s = Stack([])

        while True:
            if x:
                s.push(x)
                x = x.lc
            elif not s.empty():
                x = s.pop()
                print(x.data, end="->")
                x = x.rc
            else:
                break

    def travIn_I3(self):
        """二叉树中序遍历迭代版本3"""
        x = copy(self)
        backtrack = False

        while True:
            if not backtrack and BinNode.HasLChild(x):
                x = x.lc
            else:
                print(x.data, end="->")
                if BinNode.HasRChild(x):
                    x = x.rc
                    backtrack = False
                else:
                    x = x.succ()
                    if not x:
                        break
                    backtrack = True

    def gotoHLVEL(self, s: Stack):
        x = s.top()
        while x:
            if BinNode.HasLChild(x):
                if BinNode.HasRChild(x):
                    s.push(x.rc)
                s.push(x.lc)
            else:
                s.push(x.rc)
            s.pop()

    def travPost_I(self):
        """二叉树后序遍历迭代"""
        x = copy(self)
        s = Stack([])

        if x:
            s.push(x)
        while not s.empty():
            if s.top() != x.parent:
                self.gotoHLVEL(s)
            x = s.pop()
            print(x.data, end="->")

    def __eq__(self, other):
        if not other or not self:
            return False
        return self._data == other.data

    def __ne__(self, other):
        if not other or not self:
            return False
        return not (self._data == other.data)

    def __lt__(self, other):
        if not other or not self:
            return False
        return self._data < other.data

    def __le__(self, other):
        if not other or not self:
            return False
        return self._data <= other.data

    def __gt__(self, other):
        if not other or not self:
            return False
        return self._data > other.data

    def __ge__(self, other):
        if not other or not self:
            return False
        return self._data >= other.data

    def __repr__(self):
        return str(self._data)

    def __bool__(self):
        return bool(self._data is not None)

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
            if x.parent.parent:
                return x.parent.parent.rc
        else:
            if x.parent.parent:
                return x.parent.parent.lc
        return None

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
        self._root = root
        self._size = 0
        if self._root:
            self._size += 1

    def __bool__(self):
        return bool(self.root is not None)

    def __updateHeight(self, x: BinNode):
        x.height = 1 + max([stature(x.lc), stature(x.rc)])
        return x.height

    def __updateHeightAbove(self, x: BinNode):
        while x:
            self.__updateHeight(x)
            x = x.parent

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value: Any):
        self._size = value

    def empty(self):
        return not self._root

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, x):
        self._root = x

    def insertAsRoot(self, value: Any):
        self._size = 1
        self._root = BinNode(value)

    def insertAsLC(self, x: BinNode, value: Any):
        self._size += 1
        x.insertAsLC(value)
        self.__updateHeightAbove(x)
        return x.lc

    def insertAsRC(self, x: BinNode, value: Any):
        self._size += 1
        x.insertAsRC(value)
        self.__updateHeightAbove(x)
        return x.rc

    def attachAsLC(self, x: BinNode, S):
        """二叉树子树接入算法: 将S当作节点x的左子树接入, S本身置空"""
        assert isinstance(S, type(self))
        if not x:
            raise ValueError(f"x 不能为空节点")
        if not x.lc:
            xlc_size = 0
        else:
            xlc_size = x.lc.size()

        x.lc = S.root
        x.lc.parent = x
        self._size += (S.size - xlc_size)
        self.__updateHeightAbove(x)
        S.root = None
        S.size = 0
        del S  # 删除这个变量名
        return x

    def attachAsRC(self, x: BinNode, S):

        assert isinstance(S, type(self))
        if not x:
            raise ValueError(f"x 不能为空节点")
        if not x.lc:
            xrc_size = 0
        else:
            xrc_size = x.lc.size()

        x.rc = S.root
        x.rc.parent = x
        self._size += (S.size - xrc_size)
        self.__updateHeightAbove(x)
        S.root = None
        S.size = 0
        del S
        return x

    def remove(self, x: BinNode):
        x = BinNode.FromParentTo(x)
        xp = x.parent
        n = self.removeAt(x)
        self._size -= n

        self.__updateHeightAbove(xp)

        return n

    def removeAt(self, x: BinNode):
        """删除二叉树中位置x处的节点及其后代,返回被删除节点的数值"""
        if not x:
            return 0
        n = 1 + self.removeAt(x.lc) + self.removeAt(x.rc)
        x.data = None
        del x
        return n

    def secede(self, x: BinNode):
        """二叉树子树分离算法: 将子树x从当前树中摘除,将其封装为一颗独立子树返回"""
        if not x:
            return
        x = BinNode.FromParentTo(x)
        xp = x.parent

        if BinNode.IsRchild(x):
            xp.rc = None
        else:
            xp.lc = None

        S = BinTree(root=x)  # 新树以x为树根
        x.parent = None
        S._size = x.size()
        self._size -= S.size

        self.__updateHeightAbove(xp)

        return S

    def travLevel(self):
        if self._root:
            self._root.traLevel()

    def travPre(self):
        if self._root:
            self._root.travPre_R()

    def travIn(self):
        if self._root:
            self._root.travIn_R()

    def travPost(self):
        if self._root:
            self._root.travPost_R()

    def __repr__(self):
        print("start->", end="")
        self.travLevel()
        print("end", end='\t')
        return f"(***树大小: {self.size}:根节点: {self.root}***)"


if __name__ == '__main__':
    x = BinNode(data=5, parent=None)
    tree = BinTree(root=x)
    rc = tree.insertAsRC(x, 3)
    lc = tree.insertAsLC(x, 2)
    # print(rc)
    rlc = tree.insertAsRC(rc, 4)
    rrc = tree.insertAsLC(rc, 6)
    lrrc = tree.insertAsLC(rrc, 8)

    llc = tree.insertAsRC(lc, 7)
    lrc = tree.insertAsLC(lc, 1)
    rllc = tree.insertAsRC(llc, 9)


    # print(tree)
    # print(x.succ())
    # print(x.size())
    # print(x.traLevel())
    # print(x.travPre_R())
    # print(x.travPre_I2())

    # print(x.travIn_R())
    # print(x.travIn_I1())
    # print(x.travIn_I2())
    # print(x.travIn_I3())
    # print(BinNode.IsRoot(x))
    # print(BinNode.IsLChild(lc))
    # print(BinNode.IsRchild(rc))
    # print(BinNode.HasParent(rc))
    # print(BinNode.HasLChild(x))
    # print(BinNode.HasRChild(x))
    # print(BinNode.HasChild(x))
    # print(BinNode.HasBothChild(x))
    # print(BinNode.IsLeaf(rllc))
    # print(BinNode.sibling(lc))
    # print(BinNode.uncle(rrc))
    # print(BinNode.FromParentTo(rrc))
    #
    # print(tree)
    # print(tree.size)
    # o = BinNode(-1)
    # otree = BinTree(root=o)
    #
    # o_lc = otree.insertAsLC(o, 10)
    # o_rc = otree.insertAsRC(o, 16)
    #
    # o_llc = otree.insertAsLC(o_lc, 17)
    # o_rlc = otree.insertAsRC(o_lc, 12)
    # print(otree)
    # print(otree.size)
    # tree.attachAsRC(lrc, otree)
    # print(tree)

    # print(tree.remove(llc))
    # print(tree)

    print(tree.secede(rc))
    print(tree)

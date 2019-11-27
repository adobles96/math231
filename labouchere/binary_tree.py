class Node:
    def __init__(self, data, list_init=False):
        '''
        Initializes a Node object.

        :param data: if list_init=False this is just the data or value to be stored at the Node.
        if list_init=True this should be a tuple of two lists, the first consisting of the pre-order traversal of
        the desired tree, the second consisting of the in-order traversal of the desired tree. These two lists are then
        used to construct the specified tree with root at the present instance.

        :param list_init: See above.
        '''
        if list_init:
            preorder = data[0]
            inorder = data[1]
            assert(preorder is not None)
            self.data = preorder[0]
            i = inorder.index(self.data)

            l_preorder = preorder[1 : i + 1]
            if l_preorder:
                l_inorder = inorder[: i]
                self.left = Node((l_preorder, l_inorder), True)
            else:
                self.left = None

            r_preorder = preorder[i + 1 : ]
            if r_preorder:
                r_inorder = inorder[i + 1: ]
                self.right = Node((r_preorder, r_inorder), True)
            else:
                self.right = None
        else:
            self.data = data
            self.left = self.right = None

    def __lt__(self, other):
        return self.data < other.data

    def __str__(self):
        # Returns a pre-order traversal of the tree
        s = str(self.data)
        if self.left:
            s += " " + str(self.left)
        if self.right:
            s += " " + str(self.right)
        return s

    def __mul__(self, other):
        if type(other) is Node:
            return self.data * other.data
        else:
            return self.data * other

    def __add__(self, other):
        if type(other) is Node:
            return self.data + other.data
        else:
            return self.data + other

    def set_left(self, data):
        if type(data) is Node:
            self.left = data
        else:
            self.left = Node(data)

    def set_right(self, data):
        if type(data) is Node:
            self.right = data
        else:
            self.right = Node(data)

    def compute_bottom_up(self, fn):
        '''
        Carries out a bottom up computation according to a function fn.
        The bottom up computation refers to a computation from the leaves to the root.

        :param fn: A function that takes in the data at a Node and its two child nodes, and returns a value that
        will replace the current node's data. Its signature is given by fn(data, left, right)
        fn must handle the case where one of the inputs is None. If both inputs are None, the data of the leaf node is
        not changed.

        :return:  None. Computation is carried out in-place
        '''
        if self.left is None:
            if self.right is None:
                return
            else:
                self.right.compute_bottom_up(fn)
                self.data = fn(self.data, None, self.right)
        elif self.right is None:
            self.left.compute_bottom_up(fn)
            self.data = fn(self.data, self.left, None)
        else:
            self.left.compute_bottom_up(fn)
            self.right.compute_bottom_up(fn)
            self.data = fn(self.data, self.left, self.right)


def mul(data, left, right):
    if left is None:
        if right is None:
            return 1
        else: return right.data
    elif right is None:
        return left.data
    else: return left * right

def add(data, left, right):
    if left is None:
        if right is None:
            return 0
        else: return right.data
    elif right is None:
        return left.data
    else: return left + right

def test():
    # Tests the compute_bottom_up function with the following tree.
    #        6
    #       /  \
    #      4    9
    #       \  / \
    #        2 3  8
    #            /
    #           12
    # Desired outputs: mul: 12*3*2 = 72  -- add: 12+3+2 = 17
    root = Node(([6,4,2,9,3,8,12], [4,2,6,3,9,12,8]), True)
    root.compute_bottom_up(add)
    print(root.data)
    print(root*6)


if __name__ == '__main__':
    test()

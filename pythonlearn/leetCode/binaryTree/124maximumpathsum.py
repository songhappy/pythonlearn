# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

import sys

class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """

    def maxPathSum(self, root):
        # write your code here
        result = self.dfs(root)
        return result[0]

    def dfs(self, root):
        # exit
        if root is None:
            return -sys.maxsize, -sys.maxsize

        # devide
        left = self.dfs(root.left)
        right = self.dfs(root.right)

        # conquer
        acrossRoot = max(left[1], 0) + root.val + max(right[1], 0)
        root2Any = max(left[1], right[1], 0) + root.val
        any2Any = max(left[0], right[0], acrossRoot)

        return any2Any, root2Any

import sys
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def maxPath2(root):
    if root is None: return -sys.maxint
    left = maxPath2(root.left)
    right = maxPath2(root.right)
    return max(0, left, right) + root.val

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    """
    @param root: A Tree
    @return: Inorder in ArrayList which contains node values.
    """

    def inorderTraversal1(self, root):
        # write your code here
        if not root:
            return []
        inorder = [root.val]
        left = []
        right = []
        if root.left: left = self.inorderTraversal(root.left)
        if root.right: right = self.inorderTraversal(root.right)
        return left + inorder + right

    def inorderTraversal2(self, root):
        self.inorder = []
        self.traverse(root)
        return self.inorder

    def traverse(self, root):
        if not root:
            return
        if root.left: self.traverse(root.left)
        self.inorder.append(root.val)
        if root.right: self.traverse(root.right)
        return

    def inorderTraversal(self, root):
        stack = []
        result = []
        curr = root
        while (curr or stack):
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            result.append(curr.val)
            curr = curr.right
        return result
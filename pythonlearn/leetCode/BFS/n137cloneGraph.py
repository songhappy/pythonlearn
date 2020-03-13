# no level can be detected
class UndirectedGraphNode:
     def __init__(self, x):
         self.label = x
         self.neighbors = []



class Solution:
    """
    @param node: A undirected graph node
    @return: A undirected graph node
    """

    def cloneGraph(self, node):
        # write your code here
        if node is None: return None
        root = node
        nodes = self.getNodes(node)
        # new nodes and old-new-mapping
        mapping = {}
        for node in nodes:
            mapping[node] = UndirectedGraphNode(node.label)
        for node in nodes:
            new_node = mapping[node]
            for neighbor in node.neighbors:
                new_neighbor = mapping[neighbor]
                new_node.neighbors.append(new_neighbor)
        return mapping[root]

    def getNodes(self, node):
        nodes = set([node])
        import collections
        queue = collections.deque([node])
        while queue:
            head = queue.popleft()
            for neighbor in head.neighbors:
                if neighbor not in nodes:
                    nodes.add(neighbor)
                    queue.append(neighbor)

        return nodes



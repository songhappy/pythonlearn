# dependencies=[[a,b],[b,c],[d,e],[c,e]], n=5
# output [e,d,c,b,a]
import collections

class Solution:
    def __init__(self, dependencies):
        graph = {}
        neighbors = []
        for i in range(len(dependencies)):
            graph[dependencies[i][0]] = neighbors.append(dependencies[i][1])
        self.graph = graph

    def get_indegrees(self, graph):
        indegrees = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                indegrees[neighbor] += 1
        return indegrees

    def top_sort(self):
        indegrees = self.get_indegrees(graph=self.graph)
        start_nodes = [k for k, v in indegrees.items if v == 0]
        queue = collections.deque(start_nodes)
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in self.graph[node]:
                indegrees[neighbor] = indegrees[neighbor] - 1
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)
        return result



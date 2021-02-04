class Solution:
    """
    @param grid: a boolean 2D matrix
    @return: an integer
    """

    # 5. if visited, always into visited when into queue

    def numIslands(self, grid):
        # write your code here
        matrix = grid
        if not matrix: return 0

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.visited = set()
        count = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] and (i, j) not in self.visited:
                    self.bfs(matrix, i, j)  # update visited
                    count = count + 1
        return count

    def bfs(self, matrix, i, j):
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        from collections import deque
        self.visited.add((i, j))
        queue = deque([(i, j)])
        while queue:
            (i, j) = queue.popleft()
            for (di, dj) in neighbors:
                (new_i, new_j) = (i + di, j + dj)
                if self.is_valid(new_i, new_j, matrix):
                    queue.append((new_i, new_j))
                    self.visited.add((new_i, new_j))
        return

    def is_valid(self, i, j, matrix):
        if i >= len(matrix) or i < 0: return False
        if j >= len(matrix[0]) or j < 0: return False
        if (i, j) in self.visited: return False
        if not matrix[i][j]: return False
        return True





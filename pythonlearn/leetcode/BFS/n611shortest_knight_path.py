import collections
#Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b


class Solution:
    """
    @param grid: a chessboard included 0 (false) and 1 (true)
    @param source: a point
    @param destination: a point
    @return: the shortest path
    """

    def shortestPath(self, grid, source, destination):
        # write your code here
        distance = {
            (source.x, source.y): 0}  # (x,y):0, the distance from source to current location
        deltas = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]
        queue = collections.deque([(source.x, source.y)])
        while queue:
            (x, y) = queue.popleft()
            if (destination.x, destination.y) == (x, y):
                return distance[(x, y)]
            for (dx, dy) in deltas:
                (n_x, n_y) = (x + dx, y + dy)
                if (n_x, n_y) in distance: continue
                if not self.is_valid(n_x, n_y, grid): continue
                distance[(n_x, n_y)] = distance[(x, y)] + 1
                queue.append((n_x, n_y))

        return -1

    def is_valid(self, x, y, grid):
        if x < 0 or x >= len(grid): return False
        if y < 0 or y >= len(grid[0]): return False
        return not grid[x][y]

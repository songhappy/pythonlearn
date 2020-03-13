1. when to use BFS
    graph traversal
        level order traversal 
        connected component,  找到连通分量的个数啊, 给一个点，一圈一圈的遍历
        topological sorting
     shortest path in simple graph
     iteration solution for all possible results
2. queue.Queue is intended for allowing different threads to communicate using queued messages/data, whereas collections.deque is simply intended as a datastructure. That's why Queue.Queue has methods like put_nowait(), get_nowait(), and join(), whereas collections.deque doesn't. Queue.Queue isn't intended to be used as a collection, which is why it lacks the likes of the in operator.
    It boils down to this: if you have multiple threads and you want them to be able to communicate without the need for locks, you're looking for Queue.Queue; if you just want a queue or a double-ended queue as a datastructure, use collections.deque.
    deque.append(), deque.pop(), deque.popleft()
    queue.get(), queue.put()
3. by level VS no level
4. graph bfs different from tree bfs cause in graph you need check if it is visited or not, and {}.add(node) and queue.append(node) at the same time
    n137
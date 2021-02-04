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
    由点到面，每一次一个点，可以省一成循环
4. graph bfs different from tree bfs cause in graph you need check if it is visited or not, and {}.add(node) and queue.append(node) at the same time
    n433
5. careful when constructing a graph, could be a root node, could be matrix or dictionaries
  
6. toplogical sorted related, need build graph and calculate indegrees, and return an order, eg. n127, n615, n605
    ask : any top_order n 127
          is there any top_order n615, 611
          the only top order, len(queue) > 1, then None n605
          
7. bfs related
    ask: bfs order, return the order n137, n242
         shortest path, need update visited in bfs or return order. length in n120, distance in n611, visited in n433

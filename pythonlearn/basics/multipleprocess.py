# Queue.Queue is just an in-memory queue that knows how to deal with multiple threads using it at the same time, intended for allowing different threads to communicate using queued messages/data, It only works if both the producer and the consumer are in the same process.
# multipleprocessing.queue, for multiple processes, It uses pipes to provide a way for two processes to communicate.
# collections.deque is simply intended as a datastructure.
# It boils down to this: if you have multiple threads and you want them to be able to communicate without the need for locks, you're looking for Queue.Queue; if you just want a queue or a double-ended queue as a datastructure, use collections.deque. if you want to use multiple process to exchane data, use multipleprocessing.queue

# process vs thread
# process: an instance of a program (e.g a python interpreter)
# takes advantage of multiple CPUs and cores
# separate memory space, memory is not shared between processes
# grat for CUP-bound processing
# new process is started independently from other process
# one GIL for each process -> avoid GIL limitation
# heavyweight
# starting a process is slower than starting a thread
# more memory
# IPC(inter-process communication) is more complicated

# thread: an entity within a process that can be scheduled(known as lightweight process)
# a process can spwan multiple threads
# all threads within a process share the same memory
# starting a hread is faster than starting a process
# great for I/O-bound tasks
# threading is limited by GIL: only one thread at a time
# no effect for CPU-bound tasks
# not interuuptable/killable
# carefule with race conditions

# GIL, clobal interpreter lock
# a lock that allows only one thread at a time to execute in python
# needed in CPython because memory management is not thread-safe
# can cause memory leak when two thread access the same variable at the same time
# avoid gil
# use multiprocessing
# use a different, free-threaded python implementation(Jpython, IronPython)
# use python as a wrapper for third-party (C/C++ -> humpy, scipy)

from multiprocessing import Process, Value, Array, Lock
from threading import current_thread, Thread
from queue import Queue

import time


def worker(q, lock):
    while True:
        value = q.get()
        with lock: # without lock, multiple threads try to print the same value
            print(f'in {current_thread().name} got {value}')
        q.task_done()


def mult_thread():
    q = Queue()
    lock = Lock()
    num_threads = 10
    for i in range(num_threads):
        thread = Thread(target=worker, args=(q, lock))
        thread.daemon = True  # the thread dies when main thread dies
        thread.start()
    for i in range(1, 21):
        q.put(i)
    q.join()  # it blocks main thread untill the workers have processed everthing that is the in queue
    print('end queue show2')

def add_100(number, lock):
    for i in range(100):
        time.sleep(0.01)
        with lock:
            number.value += 1

def multprocess1():
    lock = Lock() # lock means it will lock the variable when one thread is using it and other process
    # can only access the value changed by this thread, 保证能改变数据
    shared_value = Value('i', 0)
    print("number at beginning", shared_value.value)
    p1 = Process(target=add_100, args=(shared_value, lock))
    p2 = Process(target=add_100, args=(shared_value, lock))
    p1.start()
    p2.start()  # p1, p2, will do the job at parallel
    p1.join()
    p2.join()   # we wait till processes finish
    print("number at the end", shared_value.value)
mult_thread()
multprocess1()

def square(numbers, queue):
    for i in numbers:
        queue.put(i*i)
def make_negative(numbers, queue):
    for i in numbers:
        queue.put(-1*i)

from multiprocessing import Queue
def mult_process2():
    numbers = range(1, 6)
    q = Queue()
    p1 = Process(target=square, args=(numbers, q))
    p2 = Process(target=make_negative, args=(numbers, q))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    while not q.empty():
        print(q.get())
mult_process2()

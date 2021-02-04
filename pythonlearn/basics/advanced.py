# https://www.youtube.com/playlist?list=PLqnslRFeH2UqLwzS0AwKDKLrpYBKzLBy2

import sys
from functools import reduce

# inspect
import inspect
from queue import  Queue
#print(inspect.getsource(Queue))

# magic methods/data model methods and python data model
# magic methods are double underscored methods
from queue import Queue as q
class Queue(q):
       def __repr__(self):
              return f"Queue({self.qsize()})"
       def __add__(self, other):
              self.put(other)
qu = Queue()
print(qu)
qu + 9

# metaclasses
class Foo:
       def show(self):
              print("hi")
def add_atribute(self):
       self.z = 9
# type(name, bases, dict) -> a new type
# how to use type to create a class
Test = type('Test', (Foo, ), {"x":5, "add_attribute":add_atribute})
t = Test()
t.wy = "hello"
print(t.wy)
t.show()
t.add_attribute()
print(t.z)


# meta class can change the class atribute and force in different ways in different user iterface
class Meta(type):
       def __new__(self, class_name, bases, attrs):
              a = {}
              for name, val in attrs.items():
                     if name.startswith("__"):
                            a[name] = val
                     else:
                            a[name.upper()] = val  # change atributes to upper class
              return type(class_name, bases, a)
class Dog(metaclass=Meta):
       x = 5
       y = 8
       def hello(self):
              print("hi")
d = Dog()
d.HELLO()

# decorators wrap a function, modifying its behavior
# used when you want to modify behavior of a function without toughing it or changing it, check input and output values, timer
def my_decorator(func):  # pass a function to a function
    def wrapper(*args, **kwargs): # implement wrapper to change the behavior
        print("Something is happening before the function is called.")
        rv = func(*args, **kwargs)
        print("Something is happening after the function is called.")
        return rv
    return wrapper
# say_whee = my_decorator(say_whee)
@my_decorator
def say_whee():
    print("Whee!")
say_whee()

import time
def timer(func):
       def wrapper(*args, **kwargs):
              start = time.time()
              rv = func(*args, **kwargs)
              total = time.time() - start
              print("time:", total)
              return rv
       return wrapper
@timer
def test():
       time.sleep(2)
test()


# generators
x = [i**2 for i in range(10)]  # take too much memory
def gen(n):
       for i in range(n):
              yield i**2  # pause
g = gen(10)
print(next(g))   # next is defined in builtin
print(next(g))
print(sys.getsizeof(g))
my_generator = (i for i in range(10) if i %2 ==0)
for i in my_generator:
       print(i)

print(sorted(gen(4)))
print(sum(gen(4)))

def countdown(num):
       while num > 0:
              yield num
              num -= 1
cd = countdown(4)
value = next(cd)
print(next(cd))
print(sum(countdown(4)))
print(sorted(countdown(4)))

def firstn(n):
       num = 0
       while num < n:
              yield num
              num += 1
print(sum(firstn(10)))


# context managers
# allocate and release resources precisely when you want to.
# using with, it will close this file at the end no matter break or not
with open("file.txt", "w") as file:
       file.write("hello")
# other examples, lock, lock.aqure, lock.release, with
# own context manager
from contextlib import contextmanager
@contextmanager
def open_managed_file(filename):
       f = open(filename, 'w')
       try:
              yield f
       finally:
              f.close()
with open_managed_file('file.txt') as f:
       f.write('some thing')


# random numbers
import random
random.seed(1) # reduceable
a = random.random()
a = random.randint(1, 10)

import secrets #not reduc["abcdedf"]eble
a = secrets.randbelow(10)
a = secrets.randbits(4) #largest 15, 4 bits 1111
mylyst = list("abcdedf")
print(mylyst)
a = secrets.choice(mylyst)
print(a)

import numpy as np
np.random.seed(1)  # reproduceable
a = np.random.rand(3)
print(a)
a = np.random.randint(0,10,(3,4))
print(a)
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr)
np.random.shuffle(arr)
print(arr)


# copy, shallow copy and deep copy
# Making a shallow copy of an object won’t clone child objects. Therefore, the copy is not fully independent of the original.
# A shallow copy means constructing a new collection object and then populating it with references to the child objects found in the original. In essence, a shallow copy is only one level deep. The copying process does not recurse and therefore won’t create copies of the child objects themselves.
# A deep copy makes the copying process recursive. It means first constructing a new collection object and then recursively populating it with copies of the child objects found in the original. Copying an object this way walks the whole object tree to create a fully independent clone of the original object and all of its children.
import copy
org = [0,1,2,3,4]
cpy = org  # copy, point to the same object, org will change if cpy change
cpy[0] = -10
print(org)
cpy = copy.copy(org) # shallow copy, orginal does not get affected at first level
cpy = org[:]
cpy = list(org)
cpy[0] = -10
print(org) # not affected

org = [[0,1,2,3,4],[5,6,7,8]]
cpy = copy.copy(org)
cpy[0][1] ='X'
print(cpy)
print(org) # affected, because shallow copy only has one level deep

class Person:
       def __init__(self, name, age):
           self.name = name
           self.age = age
p1 = Person('alex', 27)
p2 = copy.copy(p1)
p2.age = 28
print(p1.age)  # not affected
print(p2.age)

p2 = copy.deepcopy(p1)
p2.age = 28
print(p1.age)  # not affected
print(p2.age)

p2 = p1
p2.age = 28
print(p1.age)  # affected
print(p2.age)


# collections, Counter, namedtuple, OderedDict, deque
from collections import Counter
a = 'aaabbbbbccccccc'
my_counter = Counter(a)
print(my_counter.items())
mydict = dict(my_counter)
print(my_counter.most_common(2)[0][0])
print(list(a))

from collections import namedtuple
Point = namedtuple('Point', 'x,y') #created a class Point with fields of x and y
pt = Point(1, -4)
print(pt)
print(pt.x, pt.y)

from collections import OrderedDict # like a dict, remender the order to be inserted
keys = list('acb')
values = [1,2,3]
mydict = dict(zip(keys, values))
print(mydict)
ordered_dict = OrderedDict(mydict)
print(ordered_dict)
from collections import defaultdict
d = defaultdict(lambda : -1) # if key not found return -1
d = defaultdict(int) # if key not found, return 0
d['a'] = 1
print(d['a'])
print(d['c'])

from collections import deque
d = deque()
d.append(1)
d.append(2)
d.appendleft(3)
print(d)
d.pop()
d.popleft()
d.clear()
d.extend([5,6,7])
d.extendleft([-1,-2])
print(d)
d.rotate(2) # rotate right 2 elements to left position


#string: ordered mutable text representation
mystring = "hello world"
substring = mystring[::2] # 2 here is step, from beginning to end
print(substring)
print(mystring[::-1])
mystring.replace('world', 'universe')
mylist = mystring.strip().split()
print(mylist)
mystring2 = ' '.join(mylist)
print(mystring2)
var = 'good'
# format string, %, .format(), f-strings
mystring = "the variable is %s" % var  # f for float, d for integer  decimal value
print(mystring)
var1 = 3.1415
var2 = 5.15
mystring = "the numbers are {:.2f} and {}".format(var1, var2)  # new way to condtruct string
print(mystring)
mystring = f"the numbers are {var1} and {var2}"  # new way to condtruct string
print(mystring)
i = 10
print(f'the number is {i}, squared is {i**2}')

print ("%03d" % (1,))
print("{:03d}".format(1))
print(f"{1:.2f}")



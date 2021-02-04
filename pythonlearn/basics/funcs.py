from functools import reduce

mylist = [1, 2, 3]
mydict = {'a':1,'b':2,'c':3}

# Inside a function header:
# *args, can pass any number of positional arguments
# **kwards, can pass any number of key value arguments
def functionA(*args, **kwargs):
       print(args)
       print(kwargs)

# In a function call:
# * unpacks a list or tuple into position arguments.
# ** unpacks a dictionary into keyword arguments.
functionA(1, 2, 3, 4, 5, 6, a=2, b=3, c=5)
functionA(*mylist, **mydict)  #it is similar to functionA(1, 2, 3, 4, a=10, b=20)

def foo(a,b,c):
    print(a,b,c)
foo(*mylist) # * uppack the list, number of elements should match
foo(**mydict) # * uppack the dictionary, number of elements should match, keys should match names of parameters

def foo(a, b, *args, **kwargs):
    print(a, b)
    for arg in args:
        print(arg)
    for key in kwargs:
        print(key, kwargs[key])
foo(1, 2, 3, 4, 5, six=6, seven=7)

def foo(*args, last):
    for arg in args:
        print(arg)
    print(last)
foo(1,2,3, last=100)

mylist = [1,2,3]
foo(*mylist,last=100 ) # * uppack the list
mylist =(1, 2, 3)
foo(*mylist,last=100 ) # * uppack the tuple
mydict = {'a':1,'b':2,'c':3}
print(*mydict)
foo(*mydict,last=100) # * uppack the dictionary, only get the keys
# print(**mydict) will not work


mylist = [1, 2, 3]
mydict = {'a':1,'b':2,'c':3}
def foo(a,b,c):
    print(a,b,c)
foo(*mylist) # * uppack the list, number of elements should match
foo(**mydict) # * uppack the dictionary, number of elements should match

# more about * unpacking a seq into elements
newlist = [*mylist, 100, 200]
numbers = (1,2,3,4,5,6,7)
beginning, *middle, secondlast, last = numbers
print(beginning)
print(middle)
print(last)
myset = {7,8,9}
mytuple = (1,2,3,4,5,6)
newlist = [*myset, *mytuple]
print(newlist)
#print(type(*myset)) # failed TypeError: type.__new__() argument 1 must be str, not int
dict_a ={'a':1, 'b':2, 'c':3}
dict_b ={'d':4,'e':5,'f':6}
new_dict ={**dict_a, **dict_b}
print(new_dict)

# python call by object, object reference, parameter passed in is a reference to an object,
# but the reference is passed by value
# immutable will not change value
# mutable like list dictionaries can be changed, it will change, but if you rebind reference, outside reference will not change
def foo(mylist):
    mylist.append(3) # will change
    mylist[0] = -100
mylist = [0, 1]
foo(mylist)
print(mylist)

def foo(mylist):
    mylist = mylist + [10,20]  # rebind reference, so outside reference will not change
    mylist.append(3)
    mylist[0] = -100
mylist = [0, 1]
foo(mylist)
print(mylist)


# define a class inside a function
def build_trainer(name):
       class trainer_cls():
              def __init__(self, config):
                     pass
              def _train(self):
                     pass
       trainer_cls.__name = name
       trainer_cls.__qualname__ = name
       return trainer_cls
A2cTrainer = build_trainer("a2c") # works as a class constructor, not an object constructor
trainer = A2cTrainer(config = "blabla")

# define a function inside a function
def func1():
    def func2(x):
        return x+1
    return func2
new_func = func1()
x = (2)
print(x)

# define a function inside a function
class PolicyWithValue():
       pass
def build_policy(env, policy_network, value_work=None):
       def policy_fn(nbath=None,nstes=None):
              policy = PolicyWithValue(env=env)
              pass
              return policy
       return policy_fn()
def builder(name):
    if callable(name):
        return name
    else:
        print("wrong")

#print(builder(func3))

def sf01(a):
    return a + 1

x, y, z = 1,2,3
out = [*map(sf01,(x, y, z))]
print(out)

# lumbda function
# lambda arguments: expression, usually used only once in the code or as argument for sorted, map, filter and reduce
mult = lambda x,y: x*y
mult(5,10)
prints2D = [(1,2),(15,1),(5,-1),(10,4)]
sorted = sorted(prints2D, key=lambda x: x[0]+x[1])
# map(func, seq)
a = [1,2,3,4,5]
b = map(lambda x: x*2, a)
c = map(lambda x: x%2 ==0, a)
c = [x for x in a if x%2==0]
#reduce(func, seq)
print(list(b))
d = reduce(lambda x,y: x*y, a)
print(d)

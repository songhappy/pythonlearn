
# https://www.youtube.com/watch?v=C-gEQdGVXbk
# https://www.youtube.com/watch?v=8OKTAedgFYg

import sys
# loops
data = [1,2,-4,-3]
for idx, num in enumerate(data):
    if num < 0:
        data[idx] = 0
names = ['a', 'b','c','d']
for name, age in zip(names, data):
    print(f'{name} is actually {age}')
mydict = {'item': 'football', 'price':100}
for key, val in mydict.items():
    print(f'key is {key}, value is {val}')

# list comprehensions
squares = [i*i for i in range(10)]
filtered = [ele for ele in squares if ele%2==0]
modified = [0 if ele%2==0 else ele for ele in squares]
matrix2d = [[i*j for j in range(5)] for i in range(3)]
print(matrix2d)

# sort complex iterables with sorted
sorted_data = sorted(data, reverse=True)
data = [{'name':'Max','age':6},
        {'name':'Lisa', 'age':20}]
sorted_data = sorted(data, key=lambda x: x['age'])

# store unique values with sets
mylist = [1,2,3,4,5,2,3,4]
myset = set(mylist)

# save memory with generators
mygen = (i for i in range(10000))
print(sum(mygen))
print(sys.getsizeof(mygen), 'bytes')

# define default values in dictionaires with .get() and .setdefault()
mydict = {'item': 'football', 'price':100}
count = mydict.get('count', 0) # will return value of 0 if key not exist
count = mydict.setdefault('count', 0)

# count hashable objects with collections.Counter
from collections import Counter
counter = Counter(mylist)
print(counter)
print(counter[1])

# formate strings with f-Strings
var1,var2 = 1,2
mystring = f"the numbers are {var1} and {var2}"  # new way to condtruct string
print(mystring)
i = 10
print(f'the number is {i}, squared is {i**2}')
print ("%03d" % (1,))
print("{:03d}".format(1))
print(f"{1:.2f}")
# large number and format
num1 = 100_000_000
num2 = 10_000
total = num1 + num2
print(f'{num1 + num2:,}')
# concatenate strings with .join()

# merge tow dictionaries using **
d1 = {'item': 'football', 'price':100}
d2 = {'item': 'football', 'count': 3}
merged_dict = {**d1, **d2}
print(merged_dict)

# unpack tuple
a,b,*c = (1,2,3,4,5)
print(a)
print(c)
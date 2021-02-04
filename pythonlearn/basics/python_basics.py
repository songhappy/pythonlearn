# basic style of different sings
# space after the sign: ","
# space before: 单元运算符, "-", "!"
# space before and after + - = except when give default values in functions
# python numbers
#int, float and complex how many bytes per each number
a = 5
print(type(a))
print(isinstance(a, int))
import sys
a = [1,2,3,4]
print(sys.getsizeof(a))
print(sys.getsizeof(0))
print(sys.getsizeof(0.0))
#casting
int(3.0)
float(4)
print(str(a))

#ramdon numbers and matrix
import numpy as np
import random
random.seed(1)
a = random.random() # float [0.0, 1.0)
b = random.randint(1, 10) # int
np.random.seed(1)
am = np.random.random([2,3,4])  #am and cm just different ways to generate float numbers
cm= np.random.rand(2,3,4)
bm = np.random.randint(1, 10, [2, 3, 4])
print("********************")
print(am)
print(bm)
print(cm)

#python data types
#list based on arrayList
a = list()
a = [1,2,3,4]
b = a.copy()
a.append(5)
a.pop(0) #o(k)
a.pop() #o(1)
a.remove(2) #o(n)
a.insert(0, 9) #o(n)
a.extend([6,7]) #o(k)
a.sort(reverse=True)#o(n log n)
len(a)
int(3.0); float(4); print(str(a))


# collections.deque, based on doublely linked list
import collections
a = collections.deque([1, 2, 3, 4, 5])
a = collections.deque(maxlen=3) #it will popleft automatically when it reaches 3
a.copy()
a.append(6) #o(1)
a.appendleft(0) #o(1)
a.pop() #o(1)
a.popleft() #o(1)
a.extend([7,8,9]) #o(k)
a.remove(2) #o(n)
print(a.index(3))

#tuples, can not change elements
a = (1,2,3)
print(a.index(1)) #index of value
print(a[0])

#set, unique unordered
a=set()
a={1,2,3}
a.add(4)
print(a.difference({5}))
print(a.intersection({2,3}))
print(a.intersection_update({2,3}))
a.union({4, 6, 7})
# a.discard()
# a.remove()
# a.pop()
# a.clear()
a.copy()

#string, storaged in list
a = "helloworld"
a[-1]
a + "how are you"

#dictionary
a = dict()
print(a)
a =  {'name':'Jack', 'age': 26}
print(a.get("name"))
a["name"]
print(a.update({"sex": "female"}))
print(a)
a["address"]="cupertino"
print(a)
print(a.pop("name"))
print(a.popitem())
a.keys()
a.values()
print(a)

# read and write into files
# text files
import os
path0 = "/Users/guoqiong/intelWork/git/pythonlearn/tmp/python/test/"
if not os.path.exists(path0):
    os.makedirs(path0)

X = ["1,2,3,4,5","6,7,8,9,10"]
# open write and read:
with open(path0 +"demofile.txt", "w") as f:
    for item in X:
     f.write("%s\n" % item)

print("1--------")
with open(path0 +"demofile.txt") as f:
    X = [line.strip() for line in f]
print(X)

print("   xyz   ".strip())
print("2--------")
f = open(path0 +"demofile.txt")
X = f.read()
print(X)
print(type(X[0]))
print(len(X))
print(type(X))
print("3--------")
with open(path0 +"demofile.txt") as f:
    lines = f.readlines()
print(lines)

print("4--------")
for line in lines:
    print(line)

#read as csv into dataframe
import pandas
try:
    df= pandas.read_csv("./data/WF1_632822_20181120.txt")
except Exception as e:
    print("problem in reading files")

# dump and load objects, should be larger than the demofile1
import pickle
X_file = open(path0 +"demofile2.txt", 'rb')
#pickle.dump(X, X_file)
X = pickle.load(X_file)
print(X)

# pyspark read and write textfile
# rdd.coalesce(1).saveAsTextFile(...)
# users = sc.textFile("/tmp/movielens/ml-1m/users.dat") \
#     .map(lambda l: l.split("::")[0:4])\
#     .map(lambda l: (int(l[0]), l[1], int(l[2]), int(l[3])))\
#

#peopleDF = spark.read.json("examples/src/main/resources/people.json")

# DataFrames can be saved as Parquet files, maintaining the schema information.
# peopleDF.write.parquet("people.parquet")
#
# # Read in the Parquet file created above.
# # Parquet files are self-describing so the schema is preserved.
# # The result of loading a parquet file is also a DataFrame.
# parquetFile = spark.read.parquet("people.parquet")




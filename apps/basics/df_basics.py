# %%
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
# https://mp.weixin.qq.com/s/k6vzdHdVhGCxCBnslaD25g

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sys

input_path = "../../data/movielens/ml-1m/ratings.dat"


# filter
# agg
# time
# string
# %%


ratings = []
with open(input_path) as infile:
    for cnt, line in enumerate(infile):
        x = line.strip().split("::")
        y = list(map(lambda item: int(item), x))
        #y[3] = datetime.datetime.fromtimestamp(y[3])
        ratings.append(y)
    print("total records:", cnt)

df = pd.DataFrame(ratings, columns=['uid', 'mid', 'rate', 'timestamp'])
df.head()

df[['uid']].apply(pd.Series.value_counts).sort_values('uid')
df[['mid']].apply(pd.Series.value_counts).sort_values('mid')

df.groupby('rate').count().plot()
import numpy as np
df[['mid', 'rate']].groupby('mid').agg(np.mean)
df[['uid', 'rate']].groupby('uid').agg(np.mean)
#df["count"]=df[['uid', 'mid']].groupby('uid').count()
print(df.head(10))

rate=df['rate']
print(rate.describe())


print(df.loc[1])


print(df['uid'][0:10])

print("mean---------")
print((df["rate"].mean()))

df['date']=pd.to_datetime(df['timestamp'], unit = 's').dt.date
print(df.head())

df1 = df.sort_values(['uid', 'timestamp']).groupby(['uid', 'date'], as_index=False).agg(lambda x: list(x)).drop('timestamp', 1)

df1['sess_len'] = df1['mid'].map(len)
df1['sessid'] = range(0, len(df1))
print(df1.head(100))
print(df1.iloc[20])

length_sum = sum(df1['sess_len'])
print(length_sum)
probability = np.array(df1['sess_len'].apply(lambda x: x/length_sum))
print(probability)
import random
import numpy
for i in range(5):
    ssid= random.choices(range(len(df1)), probability)
    print(ssid[0])


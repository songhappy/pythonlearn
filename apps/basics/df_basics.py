#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import datetime

input_path = "../../data/movielens/ml-1m/ratings.dat"


# In[2]:


ratings = []
with open(input_path) as infile:
    for cnt, line in enumerate(infile):
        x = line.strip().split("::")
        y = list(map(lambda item: int(item), x))
        #y[3] = datetime.datetime.fromtimestamp(y[3])
        ratings.append(y)
    print("total records:", cnt)

df = pd.DataFrame(ratings, columns=['uid', 'mid', 'rate', 'timestamp'])


# In[19]:


df.head() 


# In[4]:


df[['uid']].apply(pd.Series.value_counts).sort_values('uid') 
df[['mid']].apply(pd.Series.value_counts).sort_values('mid')


# In[5]:


df.groupby('rate').count().plot()


# In[6]:


import numpy as np
df[['mid', 'rate']].groupby('mid').agg(np.mean)


# In[7]:


df[['uid', 'rate']].groupby('uid').agg(np.mean)


# In[8]:


rate=df['rate']
rate.describe()


# In[9]:


df.loc[1]


# In[10]:


df['uid'][0:10]


# In[10]:


get_ipython().run_cell_magic('time', '', "df1user = df[df['uid']==1]\ndf1user.head()")


# In[73]:


df['date']=pd.to_datetime(df['timestamp'], unit = 's').dt.date
df.head()


# In[93]:


df1 = df.sort_values(['uid', 'timestamp']).groupby(['uid', 'date'], as_index=False).agg(lambda x: list(x)).drop('timestamp', 1)

df1['sess_len'] = df1['mid'].map(len)
df1['sessid'] = range(1, len(df1) +1)
df1.head()


# In[61]:


df1[['count']].plot()


# In[68]:


df1['count'].describe()


# In[90]:


df1.iloc[20]


# In[ ]:





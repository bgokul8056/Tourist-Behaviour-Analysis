
# coding: utf-8

# In[24]:

import csv
import nltk
import numpy as np
import pandas as pd


# In[25]:

location="C:\YFCC_Dataset\yfcc_5M.csv "
data = pd.read_csv(location, header=None)


# In[26]:

data[12].head()


# In[27]:

df=data[(data[12] > 113.157332) & (data[12] < 153.918610) & (data[13] < -10.768556) & (data[13] > -43.735146)]
#AUS_Coord
 


# In[28]:

len(df)


# In[29]:

df.to_csv("C:\YFCC_Dataset\Filtered5M_AUS.csv",header=False,index=False)


# In[ ]:




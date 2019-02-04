
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error
get_ipython().magic('matplotlib inline')


# In[ ]:




# In[2]:

location="C:\YFCC_Dataset\SanFranciscoDate.csv"
df = pd.read_csv(location, header=None)


# In[3]:

df2=pd.to_datetime(df[5])


# In[4]:

len(df)


# In[5]:

months = list(range(1,13))
df[5]=pd.to_datetime(df[5])
ct=[]

for i in months:
    ct.append(len(df.loc[(df[5].dt.month == i ) ] ) )


# In[6]:

for i in range(len(ct)):
    ct[i]=ct[i]/10


# In[7]:

months=np.asarray(months)


# In[8]:

count = np.asarray(ct)


# In[9]:

import sklearn
import sklearn.linear_model as lm
get_ipython().magic('matplotlib inline')


# In[10]:

plt.scatter(months,count)
plt.plot(months,count, color='blue', linewidth=1)


# In[11]:

months = list(range(1,13))
year = list(range(2004,2014))
allmonths={}
for i in months:
    allmonths[i]=[]
for i in year:
    for j in months:
        allmonths[j].append(len(df.loc[operator.and_( df[5].dt.year == i , df[5].dt.month == j ) ] ) )


# In[12]:

allmonths


# In[27]:


monthCt=np.asarray(allmonths[4])
yr=np.asarray(year)

plt.scatter(year,monthCt)

poly = make_pipeline(PolynomialFeatures(3),regr)

poly.fit(yr.reshape(-1,1) , monthCt)

Y_pred=poly.predict(yr.reshape(-1,1).reshape(-1,1))

plt.plot(year,monthCt, color="orange" ,linewidth=1 )
plt.plot(year,Y_pred.reshape(-1,1), color='blue', linewidth=1 )

apr_diff=[]
for i in range(len(monthCt)):
    apr_diff.append(abs(monthCt[i]-Y_pred[i]))
month_diff.append(apr_diff)
sc.append(mean_absolute_error(monthCt , Y_pred))

plt.title("APRIL")
# mean_absolute_error(monthCt , Y_pred)


# In[13]:

from sklearn import linear_model as lm
regr = lm.LinearRegression()
sc=[]
monthCt=np.asarray(allmonths[1])
yr=np.asarray(year)

plt.scatter(year,monthCt)

poly = make_pipeline(PolynomialFeatures(2),regr)

poly.fit(yr.reshape(-1,1) , monthCt)

Y_pred=poly.predict(yr.reshape(-1,1).reshape(-1,1))

plt.plot(year,monthCt, color="orange" ,linewidth=1 )
plt.plot(year,Y_pred.reshape(-1,1), color='blue', linewidth=1 )

month_diff=[]
jan_diff=[]
for i in range(len(monthCt)):
    jan_diff.append(abs(monthCt[i]-Y_pred[i]))
month_diff.append(jan_diff)
    
sc.append(mean_absolute_error(monthCt , Y_pred))

plt.title("JANUARY")
mean_absolute_error(monthCt , Y_pred)


# In[14]:


monthCt=np.asarray(allmonths[2])
yr=np.asarray(year)

plt.scatter(year,monthCt)

poly = make_pipeline(PolynomialFeatures(3),regr)

poly.fit(yr.reshape(-1,1) , monthCt)

Y_pred=poly.predict(yr.reshape(-1,1).reshape(-1,1))

plt.plot(year,monthCt, color="orange" ,linewidth=1 )
plt.plot(year,Y_pred.reshape(-1,1), color='blue', linewidth=1 )

feb_diff=[]
for i in range(len(monthCt)):
    feb_diff.append(abs(monthCt[i]-Y_pred[i]))
month_diff.append(feb_diff)    
sc.append(mean_absolute_error(monthCt , Y_pred))

plt.title("FEBRUARY")
mean_absolute_error(monthCt , Y_pred)


# In[15]:


monthCt=np.asarray(allmonths[3])
yr=np.asarray(year)

plt.scatter(year,monthCt)

poly = make_pipeline(PolynomialFeatures(3),regr)

poly.fit(yr.reshape(-1,1) , monthCt)

Y_pred=poly.predict(yr.reshape(-1,1).reshape(-1,1))

plt.plot(year,monthCt, color="orange" ,linewidth=1 )
plt.plot(year,Y_pred.reshape(-1,1), color='blue', linewidth=1 )

mar_diff=[]
for i in range(len(monthCt)):
    mar_diff.append(abs(monthCt[i]-Y_pred[i]))
month_diff.append(mar_diff)    
sc.append(mean_absolute_error(monthCt , Y_pred))

plt.title("MARCH")
mean_absolute_error(monthCt , Y_pred)


# In[17]:


monthCt=np.asarray(allmonths[5])
yr=np.asarray(year)

plt.scatter(year,monthCt)

poly = make_pipeline(PolynomialFeatures(3),regr)

poly.fit(yr.reshape(-1,1) , monthCt)

Y_pred=poly.predict(yr.reshape(-1,1).reshape(-1,1))

plt.plot(year,monthCt, color="orange" ,linewidth=1 )
plt.plot(year,Y_pred.reshape(-1,1), color='blue', linewidth=1 )

may_diff=[]
for i in range(len(monthCt)):
    may_diff.append(abs(monthCt[i]-Y_pred[i]))
month_diff.append(may_diff)
sc.append(mean_absolute_error(monthCt , Y_pred))

plt.title("MAY")
mean_absolute_error(monthCt , Y_pred)


# In[18]:


monthCt=np.asarray(allmonths[6])
yr=np.asarray(year)

plt.scatter(year,monthCt)

poly = make_pipeline(PolynomialFeatures(3),regr)

poly.fit(yr.reshape(-1,1) , monthCt)

Y_pred=poly.predict(yr.reshape(-1,1).reshape(-1,1))

plt.plot(year,monthCt, color="orange" ,linewidth=1 )
plt.plot(year,Y_pred.reshape(-1,1), color='blue', linewidth=1 )

jun_diff=[]
for i in range(len(monthCt)):
    jun_diff.append(abs(monthCt[i]-Y_pred[i]))
month_diff.append(jun_diff)
sc.append(mean_absolute_error(monthCt , Y_pred))

plt.title("JUNE")
mean_absolute_error(monthCt , Y_pred)


# In[ ]:




# In[19]:


monthCt=np.asarray(allmonths[7])
yr=np.asarray(year)

plt.scatter(year,monthCt)

poly = make_pipeline(PolynomialFeatures(3),regr)

poly.fit(yr.reshape(-1,1) , monthCt)

Y_pred=poly.predict(yr.reshape(-1,1).reshape(-1,1))

plt.plot(year,monthCt, color="orange" ,linewidth=1 )
plt.plot(year,Y_pred.reshape(-1,1), color='blue', linewidth=1 )

jul_diff=[]
for i in range(len(monthCt)):
    jul_diff.append(abs(monthCt[i]-Y_pred[i]))
month_diff.append(jul_diff)   
sc.append(mean_absolute_error(monthCt , Y_pred))

plt.title("JULY")
mean_absolute_error(monthCt , Y_pred)


# In[20]:


monthCt=np.asarray(allmonths[8])
yr=np.asarray(year)

plt.scatter(year,monthCt)

poly = make_pipeline(PolynomialFeatures(3),regr)

poly.fit(yr.reshape(-1,1) , monthCt)

Y_pred=poly.predict(yr.reshape(-1,1).reshape(-1,1))

plt.plot(year,monthCt, color="orange" ,linewidth=1 )
plt.plot(year,Y_pred.reshape(-1,1), color='blue', linewidth=1 )

aug_diff=[]
for i in range(len(monthCt)):
    aug_diff.append(abs(monthCt[i]-Y_pred[i]))
month_diff.append(aug_diff)
    
sc.append(mean_absolute_error(monthCt , Y_pred))
plt.title("AUGUST")
mean_absolute_error(monthCt , Y_pred)


# In[21]:

monthCt=np.asarray(allmonths[9])
yr=np.asarray(year)

plt.scatter(year,monthCt)

poly = make_pipeline(PolynomialFeatures(3),regr)

poly.fit(yr.reshape(-1,1) , monthCt)

Y_pred=poly.predict(yr.reshape(-1,1).reshape(-1,1))

plt.plot(year,monthCt, color="orange" ,linewidth=1 )
plt.plot(year,Y_pred.reshape(-1,1), color='blue', linewidth=1 )

sep_diff=[]
for i in range(len(monthCt)):
    sep_diff.append(abs(monthCt[i]-Y_pred[i]))
month_diff.append(sep_diff)
sc.append(mean_absolute_error(monthCt , Y_pred))

plt.title("SEPTEMBER")
mean_absolute_error(monthCt , Y_pred)


# In[22]:

monthCt=np.asarray(allmonths[10])
yr=np.asarray(year)

plt.scatter(year,monthCt)

poly = make_pipeline(PolynomialFeatures(3),regr)

poly.fit(yr.reshape(-1,1) , monthCt)

Y_pred=poly.predict(yr.reshape(-1,1).reshape(-1,1))

plt.plot(year,monthCt, color="orange" ,linewidth=1 )
plt.plot(year,Y_pred.reshape(-1,1), color='blue', linewidth=1 )

oct_diff=[]
for i in range(len(monthCt)):
    oct_diff.append(abs(monthCt[i]-Y_pred[i]))
month_diff.append(oct_diff)
sc.append(mean_absolute_error(monthCt , Y_pred))

plt.title("OCTOBER")
mean_absolute_error(monthCt , Y_pred)


# In[23]:


monthCt=np.asarray(allmonths[11])
yr=np.asarray(year)

plt.scatter(year,monthCt)

poly = make_pipeline(PolynomialFeatures(3),regr)

poly.fit(yr.reshape(-1,1) , monthCt)

Y_pred=poly.predict(yr.reshape(-1,1).reshape(-1,1))

plt.plot(year,monthCt, color="orange" ,linewidth=1 )
plt.plot(year,Y_pred.reshape(-1,1), color='blue', linewidth=1 )

nov_diff=[]
for i in range(len(monthCt)):
    nov_diff.append(abs(monthCt[i]-Y_pred[i]))
month_diff.append(nov_diff)
sc.append(mean_absolute_error(monthCt , Y_pred))
plt.title("NOVEMBER")
mean_absolute_error(monthCt , Y_pred)


# In[24]:


monthCt=np.asarray(allmonths[12])
yr=np.asarray(year)

plt.scatter(year,monthCt)

poly = make_pipeline(PolynomialFeatures(3),regr)

poly.fit(yr.reshape(-1,1) , monthCt)

Y_pred=poly.predict(yr.reshape(-1,1).reshape(-1,1))

plt.plot(year,monthCt, color="orange" ,linewidth=1 )
plt.plot(year,Y_pred.reshape(-1,1), color='blue', linewidth=1 )

dec_diff=[]
for i in range(len(monthCt)):
    dec_diff.append(abs(monthCt[i]-Y_pred[i]))
month_diff.append(dec_diff)
sc.append(mean_absolute_error(monthCt , Y_pred))
plt.title("DECEMBER")
mean_absolute_error(monthCt , Y_pred)


# In[25]:

fig = plt.figure(figsize=(10, 5))
for i in range(1,13):
    a = np.empty(10)
    a.fill(i)
    plt.scatter(a,np.asarray(month_diff[i-1]))
    
plt.scatter(months,np.asarray(sc))
plt.xlabel("Months")
plt.ylabel("Seasonal Component")

plt.plot(months,np.asarray(sc), color='blue', linewidth=3)


# In[ ]:




# In[26]:

fig = plt.figure(figsize=(10, 5))
plt.scatter(months,np.asarray(sc))
plt.xlabel("Months")
plt.ylabel("Seasonal Component")
plt.plot(months,np.asarray(sc), color='blue', linewidth=1)


# In[ ]:




# In[ ]:




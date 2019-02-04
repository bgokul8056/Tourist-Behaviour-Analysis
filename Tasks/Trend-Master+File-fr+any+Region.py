
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


# In[2]:

df=pd.read_csv("C:\\YFCC_Dataset\\haleeurope.csv",header=None)


# In[3]:

df2=pd.to_datetime(df[5])
# df[5]


# In[ ]:

len(df)


# ### X_my - [month,year] , ct - no of tourists

# In[ ]:

# len(df.loc[df2.dt.year == 2015])


# In[ ]:

months = list(range(1,13))
year = list(range(2004,2014))
df[5]=pd.to_datetime(df[5])
ct=[]
#my=[]
X_my=[]
for i in year:
    for j in months:
        #s=str(j)+"-"+str(i)
        l=[j,i]
        X_my.append(l)
        #my.append(s)
        ct.append(len(df.loc[operator.and_( df[5].dt.year == i , df[5].dt.month == j ) ] ) )


# In[ ]:




# In[ ]:

months = list(range(1,13))

df[5]=pd.to_datetime(df[5])
ct_test=[]
#my=[]
X_my_test=[]
i=2014
for j in months:
    l=[j,i]
    X_my_test.append(l)
   #my.append(s)
    ct_test.append(len(df.loc[operator.and_( df[5].dt.year == i , df[5].dt.month == j ) ] ) )


# In[ ]:

X_my = np.asarray(X_my)
Ct = np.asarray(ct).reshape(-1,1)
X_my_test = np.asarray(X_my_test)
Ct_test = np.asarray(ct_test).reshape(-1,1)


# In[ ]:

X_my.shape


# In[ ]:

Ct.shape


# ### nry - normalised month year , eg: 2000.5 is Jun 2000

# In[ ]:

X_my
# NORMALISNG MONTH-YEAR VALUE
nry=[]
for i in X_my:
    g=i[0]
    k=i[1]
    norm=k+(round((g/13),2))
    nry.append(norm)


# In[ ]:

nry_test=[]
for i in X_my_test:
    g=i[0]
    k=i[1]
    norm=k+(round((g/13),2))
    nry_test.append(norm)


# # Linear - deg 1 and 2

# In[ ]:

from sklearn import linear_model as lm
regr = lm.LinearRegression()


# In[ ]:

poly = make_pipeline(PolynomialFeatures(1),regr)


# ### nry vs no of tourists

# In[ ]:

poly.fit(np.asarray(nry).reshape(-1,1) , Ct)


# In[ ]:

Y_pred=poly.predict(np.asarray(nry).reshape(-1,1))


# In[ ]:

plt.plot(np.asarray(nry),Y_pred.reshape(-1,1))
plt.plot(np.asarray(nry),Ct)


# In[ ]:

mean_absolute_error(Ct , Y_pred)


# In[ ]:




# In[ ]:

poly = make_pipeline(PolynomialFeatures(3),regr)


# In[ ]:

poly.fit(np.asarray(nry).reshape(-1,1),Ct)


# In[ ]:

Y_pred=poly.predict(np.asarray(nry).reshape(-1,1))


# In[ ]:

plt.plot(np.asarray(nry),Y_pred)
plt.plot(np.asarray(nry),Ct)


# In[ ]:

mean_absolute_error(Ct , Y_pred)


# In[ ]:




# In[ ]:

from sklearn import linear_model as lm
regr = lm.LinearRegression()


# In[ ]:

poly = make_pipeline(PolynomialFeatures(1),regr)


# ### X_my  vs  No of tourists 

# In[ ]:

poly.fit(X_my, Ct)


# In[ ]:

Y_pred=poly.predict(X_my)


# In[ ]:

plt.plot(np.asarray(nry),Y_pred.reshape(-1,1))
plt.plot(np.asarray(nry),Ct)


# In[ ]:

mean_absolute_error(Ct , Y_pred)


# In[ ]:




# In[ ]:

poly = make_pipeline(PolynomialFeatures(3),regr)


# In[ ]:

poly.fit(X_my,Ct)


# In[ ]:

Y_pred=poly.predict(X_my)


# In[ ]:

plt.plot(np.asarray(nry),Y_pred)
plt.plot(np.asarray(nry),Ct)


# In[ ]:

mean_absolute_error(Ct , Y_pred)


# In[ ]:

# from sklearn import linear_model
# clf = linear_model.LinearRegression()


# # Ridge - deg 1 and 2

# In[ ]:




# In[ ]:

poly = make_pipeline(PolynomialFeatures(1),Ridge())


# ### nry vs no of tourists

# In[ ]:




# In[ ]:

poly.fit(np.asarray(nry).reshape(-1,1) , Ct)


# In[ ]:

Y_pred=poly.predict(np.asarray(nry).reshape(-1,1))


# In[ ]:

plt.plot(np.asarray(nry),Y_pred.reshape(-1,1))
plt.plot(np.asarray(nry),Ct)


# In[ ]:

mean_absolute_error(Ct , Y_pred)


# In[ ]:




# In[ ]:

poly = make_pipeline(PolynomialFeatures(3),Ridge())


# In[ ]:

poly.fit(np.asarray(nry).reshape(-1,1),Ct)


# In[ ]:

Y_pred=poly.predict(np.asarray(nry).reshape(-1,1))


# In[ ]:

plt.plot(np.asarray(nry),Y_pred)
plt.plot(np.asarray(nry),Ct)


# In[ ]:

mean_absolute_error(Ct , Y_pred)


# In[ ]:




# In[ ]:

from sklearn import linear_model as lm
regr = lm.LinearRegression()


# In[ ]:

poly = make_pipeline(PolynomialFeatures(1),Ridge())


# ### X_my  vs  No of tourists 

# In[ ]:

poly.fit(X_my, Ct)


# In[ ]:

Y_pred=poly.predict(X_my)


# In[ ]:

plt.plot(np.asarray(nry),Y_pred.reshape(-1,1))
plt.plot(np.asarray(nry),Ct)


# In[ ]:

mean_absolute_error(Ct , Y_pred)


# In[ ]:




# In[ ]:

poly = make_pipeline(PolynomialFeatures(3),Ridge())


# In[ ]:

poly.fit(X_my,Ct)


# In[ ]:

Y_pred=poly.predict(X_my)


# In[ ]:

plt.plot(np.asarray(nry),Y_pred)
plt.plot(np.asarray(nry),Ct)


# In[ ]:

mean_absolute_error(Ct , Y_pred)


# In[ ]:




# In[ ]:




# # Lasso - deg 1 and 2

# In[ ]:

from sklearn import linear_model as lm
clf = lm.Lasso(alpha=0.1)


# In[ ]:

poly = make_pipeline(PolynomialFeatures(1),clf)


# ### nry vs no of tourists

# In[ ]:

poly.fit(np.asarray(nry).reshape(-1,1) , Ct)


# In[ ]:

Y_pred=poly.predict(np.asarray(nry).reshape(-1,1))


# In[ ]:

plt.plot(np.asarray(nry),Y_pred.reshape(-1,1))
plt.plot(np.asarray(nry),Ct)


# In[ ]:

mean_absolute_error(Ct , Y_pred)


# In[ ]:




# In[ ]:

poly = make_pipeline(PolynomialFeatures(3),clf)


# In[ ]:

poly.fit(np.asarray(nry).reshape(-1,1),Ct)


# In[ ]:

Y_pred=poly.predict(np.asarray(nry).reshape(-1,1))


# In[ ]:

plt.plot(np.asarray(nry),Y_pred)
plt.plot(np.asarray(nry),Ct)


# In[ ]:

mean_absolute_error(Ct , Y_pred)


# In[ ]:




# In[ ]:




# ### X_my  vs  No of tourists 

# In[ ]:

poly = make_pipeline(PolynomialFeatures(1),clf)


# In[ ]:

poly.fit(X_my, Ct)


# In[ ]:

Y_pred=poly.predict(X_my)


# In[ ]:

plt.plot(np.asarray(nry),Y_pred.reshape(-1,1))
plt.plot(np.asarray(nry),Ct)


# In[ ]:

mean_absolute_error(Ct , Y_pred)


# In[ ]:




# In[ ]:

poly = make_pipeline(PolynomialFeatures(3),clf)


# In[ ]:

poly.fit(X_my,Ct)


# In[ ]:

Y_pred=poly.predict(X_my)


# In[ ]:

plt.plot(np.asarray(nry),Y_pred)
plt.plot(np.asarray(nry),Ct)


# In[ ]:

mean_absolute_error(Ct , Y_pred)


# In[ ]:

# from sklearn import linear_model
# clf = linear_model.LinearRegression()


# In[ ]:




# In[ ]:

# # normalised month-year vs no of tourists - lasso
# lass= clf.fit( np.asarray(nry).reshape(-1,1) , Ct )
# Ct_pred=lass.predict(np.asarray(nry_test).reshape(-1,1))


# In[ ]:




# In[ ]:

# # normalised month-year vs no of tourists - lasso
# plt.plot(np.asarray(nry_test).reshape(-1,1), Ct , color='black', linewidth=1)
# plt.plot(np.asarray(nry_test).reshape(-1,1), Ct_pred, color='blue', linewidth=1)


# In[ ]:

# mean_absolute_error(Ct_test,Ct_pred)


# In[ ]:




# In[ ]:




# In[ ]:

# nry_sq=np.asarray(nry)**2


# In[ ]:

# # normalised month-year_squared vs no of tourists - lasso
# lass = clf.fit( nry_sq.reshape(-1,1) , Ct )
# Ct_pred_sq = lass.predict( nry_sq.reshape(-1,1))


# In[ ]:

# plt.plot(np.asarray(nry).reshape(-1,1) , Ct , color='black', linewidth=1)
# plt.plot(np.asarray(nry).reshape(-1,1) , Ct_pred_sq, color='blue', linewidth=1)


# In[ ]:

# mean_absolute_error(Ct,Ct_pred_sq)


# In[ ]:




# In[ ]:

# # month-year vs no of tourists - lasso
# lass = clf.fit( np.asarray(X_my) , Ct )
# Ct_pred_my = lass.predict( np.asarray(X_my) )


# In[ ]:

# plt.plot(nry , Ct , color='black', linewidth=1)
# plt.plot(nry , Ct_pred_my, color='blue', linewidth=1)


# In[ ]:

# mean_absolute_error(Ct,Ct_pred_my)


# In[ ]:




# In[ ]:

# nry_x2_x=np.column_stack((nry_sq,nry))


# In[ ]:

# # normalised-month-year-squared , normalised-month-year  VS no of tourists - lasso
# lass = clf.fit( nry_x2_x , Ct )
# Ct_pred_x2_x = lass.predict( nry_x2_x )


# In[ ]:

# mean_absolute_error(Ct,Ct_pred_x2_x)


# In[ ]:




# In[ ]:

# nry_cu= np.asarray(nry)**3


# In[ ]:

# # normalised-month-year-squared , normalised-month-year  VS no of tourists - lasso
# lass = clf.fit( nry_cu.reshape(-1,1) , Ct )
# Ct_pred_x3 = lass.predict( nry_cu.reshape(-1,1) )


# In[ ]:

# mean_absolute_error(Ct,Ct_pred_x3)


# In[ ]:




# In[ ]:

# X_my_sq=[]
# for l in X_my :
#     a=l[0]
#     b=l[1]
#     sq=[a**2,b**2]
#     X_my_sq.append(sq)


# In[ ]:

# X_my_sq=np.asarray(X_my_sq)


# In[ ]:

# # month-year vs no of tourists - lasso
# lass = clf.fit(X_my_sq , Ct )
# Ct_pred_Xmy = lass.predict(X_my_sq)


# In[ ]:

# mean_absolute_error(Ct,Ct_pred_Xmy)


# In[ ]:




# In[ ]:




# In[ ]:




import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error
%matplotlib inline
from sklearn import linear_model as lm
regr = lm.LinearRegression()
clf = lm.Lasso(alpha=0.1)

#Loading CSV corresponding to a particular region
df=pd.read_csv("C:\\YFCC_Dataset\\haleeurope.csv",header=None)
df2=pd.to_datetime(df[5])
months = list(range(1,13))
year = list(range(2004,2014))
df[5]=pd.to_datetime(df[5])
ct=[]

X_my=[]
for i in year:
    for j in months:
        l=[j,i]
        X_my.append(l)
        ct.append(len(df.loc[operator.and_( df[5].dt.year == i , df[5].dt.month == j ) ] ) )

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
X_my = np.asarray(X_my)
Ct = np.asarray(ct).reshape(-1,1)
X_my_test = np.asarray(X_my_test)
Ct_test = np.asarray(ct_test).reshape(-1,1)
X_my

# NORMALISNG MONTH-YEAR VALUE
nry=[]
for i in X_my:
    g=i[0]
    k=i[1]
    norm=k+(round((g/13),2))
    nry.append(norm)

nry_test=[]
for i in X_my_test:
    g=i[0]
    k=i[1]
    norm=k+(round((g/13),2))
    nry_test.append(norm)

#Linear - deg 1 and 2
poly = make_pipeline(PolynomialFeatures(1),regr)

#nry vs no of tourists
#Trying out various combinations

poly.fit(np.asarray(nry).reshape(-1,1) , Ct)
Y_pred=poly.predict(np.asarray(nry).reshape(-1,1))
plt.plot(np.asarray(nry),Y_pred.reshape(-1,1))
plt.plot(np.asarray(nry),Ct)
mean_absolute_error(Ct , Y_pred)

poly = make_pipeline(PolynomialFeatures(3),regr)
poly.fit(np.asarray(nry).reshape(-1,1),Ct)
Y_pred=poly.predict(np.asarray(nry).reshape(-1,1))
plt.plot(np.asarray(nry),Y_pred)
plt.plot(np.asarray(nry),Ct)
mean_absolute_error(Ct , Y_pred)

poly = make_pipeline(PolynomialFeatures(1),regr)

#X_my vs No of tourists

poly.fit(X_my, Ct)
Y_pred=poly.predict(X_my)
plt.plot(np.asarray(nry),Y_pred.reshape(-1,1))
plt.plot(np.asarray(nry),Ct)
mean_absolute_error(Ct , Y_pred)

poly = make_pipeline(PolynomialFeatures(3),regr)
poly.fit(X_my,Ct)
Y_pred=poly.predict(X_my)
plt.plot(np.asarray(nry),Y_pred)
plt.plot(np.asarray(nry),Ct)
mean_absolute_error(Ct , Y_pred)

#Ridge - deg 1 and 2
poly = make_pipeline(PolynomialFeatures(1),Ridge())

#nry vs no of tourists

poly.fit(np.asarray(nry).reshape(-1,1) , Ct)
Y_pred=poly.predict(np.asarray(nry).reshape(-1,1))
plt.plot(np.asarray(nry),Y_pred.reshape(-1,1))
plt.plot(np.asarray(nry),Ct)
mean_absolute_error(Ct , Y_pred)

​
poly = make_pipeline(PolynomialFeatures(3),Ridge())
poly.fit(np.asarray(nry).reshape(-1,1),Ct)
Y_pred=poly.predict(np.asarray(nry).reshape(-1,1))
plt.plot(np.asarray(nry),Y_pred)
plt.plot(np.asarray(nry),Ct)
mean_absolute_error(Ct , Y_pred)

poly = make_pipeline(PolynomialFeatures(1),Ridge())

#X_my vs No of tourists
poly.fit(X_my, Ct)
Y_pred=poly.predict(X_my)
plt.plot(np.asarray(nry),Y_pred.reshape(-1,1))
plt.plot(np.asarray(nry),Ct)
mean_absolute_error(Ct , Y_pred)

​poly = make_pipeline(PolynomialFeatures(3),Ridge())
poly.fit(X_my,Ct)
Y_pred=poly.predict(X_my)
plt.plot(np.asarray(nry),Y_pred)
plt.plot(np.asarray(nry),Ct)
mean_absolute_error(Ct , Y_pred)

​#Lasso - deg 1 and 2
poly = make_pipeline(PolynomialFeatures(1),clf)

#nry vs no of tourists

poly.fit(np.asarray(nry).reshape(-1,1) , Ct)
Y_pred=poly.predict(np.asarray(nry).reshape(-1,1))
plt.plot(np.asarray(nry),Y_pred.reshape(-1,1))
plt.plot(np.asarray(nry),Ct)
mean_absolute_error(Ct , Y_pred)

poly = make_pipeline(PolynomialFeatures(3),clf)
poly.fit(np.asarray(nry).reshape(-1,1),Ct)
Y_pred=poly.predict(np.asarray(nry).reshape(-1,1))
plt.plot(np.asarray(nry),Y_pred)
plt.plot(np.asarray(nry),Ct)
mean_absolute_error(Ct , Y_pred)

​#X_my vs No of tourists

poly = make_pipeline(PolynomialFeatures(1),clf)

poly.fit(X_my, Ct)
Y_pred=poly.predict(X_my)
plt.plot(np.asarray(nry),Y_pred.reshape(-1,1))
plt.plot(np.asarray(nry),Ct)
mean_absolute_error(Ct , Y_pred)
poly = make_pipeline(PolynomialFeatures(3),clf)

poly.fit(X_my,Ct)
Y_pred=poly.predict(X_my)
plt.plot(np.asarray(nry),Y_pred)
plt.plot(np.asarray(nry),Ct)
mean_absolute_error(Ct , Y_pred)
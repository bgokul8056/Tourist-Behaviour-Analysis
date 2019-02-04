import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error
%matplotlib inline
import sklearn
import sklearn.linear_model as lm

#Opening CSV File corresponding to a particular region
location="C:\YFCC_Dataset\SanFranciscoDate.csv"
df = pd.read_csv(location, header=None)
df2=pd.to_datetime(df[5])
months = list(range(1,13))
df[5]=pd.to_datetime(df[5])
ct=[]
#Finding data wrt to each Month
for i in months:
    ct.append(len(df.loc[(df[5].dt.month == i ) ] ) )
for i in range(len(ct)):
    ct[i]=ct[i]/10

months=np.asarray(months)
count = np.asarray(ct)
plt.scatter(months,count)
plt.plot(months,count, color='blue', linewidth=1)
months = list(range(1,13))
year = list(range(2004,2014))
allmonths={}

for i in months:
    allmonths[i]=[]
for i in year:
    for j in months:
        allmonths[j].append(len(df.loc[operator.and_( df[5].dt.year == i , df[5].dt.month == j ) ] ) )

#Checking value for April



regr = lm.LinearRegression()
sc=[]
#Plotting data for each month to find Trend for that particular month

#January
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

#February
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

#March
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

#April
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

#May
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

#June
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

#July
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

#August
monthCt=np.asarray(allmonths[8])
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

plt.title("AUGUST")
mean_absolute_error(monthCt , Y_pred)

#September
monthCt=np.asarray(allmonths[9])
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

plt.title("SEPTEMBER")
mean_absolute_error(monthCt , Y_pred)

#October
monthCt=np.asarray(allmonths[10])
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

plt.title("OCTOBER")
mean_absolute_error(monthCt , Y_pred)

#November
monthCt=np.asarray(allmonths[11])
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

plt.title("NOVEMBER")
mean_absolute_error(monthCt , Y_pred)

#December
monthCt=np.asarray(allmonths[12])
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

plt.title("DECEMBER")
mean_absolute_error(monthCt , Y_pred)

#Plot of Months with average no. of visitors
fig = plt.figure(figsize=(10, 5))
for i in range(1,13):
    a = np.empty(10)
    a.fill(i)
    plt.scatter(a,np.asarray(month_diff[i-1]))
    
plt.scatter(months,np.asarray(sc))
plt.xlabel("Months")
plt.ylabel("Average count of tourists")

plt.plot(months,np.asarray(sc), color='blue', linewidth=3)

fig = plt.figure(figsize=(10, 5))
plt.scatter(months,np.asarray(sc))
plt.xlabel("Months")
plt.ylabel("Seasonal Component")
plt.plot(months,np.asarray(sc), color='blue', linewidth=1)


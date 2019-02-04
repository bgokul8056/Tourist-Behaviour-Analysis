import csv
import nltk
import numpy as np
import pandas as pd

location="C:\YFCC_Dataset\yfcc_5M.csv "
data = pd.read_csv(location, header=None)

#Mention regions Coordinates within the data fields. The coordinates used here are for the Australia region
df=data[(data[12] > 113.157332) & (data[12] < 153.918610) & (data[13] < -10.768556) & (data[13] > -43.735146)]
#Convert it into a CSV file
df.to_csv("C:\YFCC_Dataset\Filtered5M_AUS.csv",header=False,index=False) 
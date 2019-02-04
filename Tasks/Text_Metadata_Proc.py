import csv
import nltk
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize,RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import os

# Specify directory and convert to CSV
location="C:\YFCC_Dataset\Filtered1M.csv"
data = pd.read_csv(location, header=None)
data.to_csv("C:\YFCC_Dataset\TP1M.csv",columns=[0,1,3,5,12,13,14],index=False,header=['LineNo','PhotoID','UserID','DateTaken','Latitude','Longitude','Accuracy'])

# Change file name
location="C:\YFCC_Dataset\Filtered1M.csv"
dest = "C:\YFCC_Dataset\TP1M.csv"
data = pd.read_csv(location, header=None)
tokenizer = RegexpTokenizer(r'[a-zA-Z_]+')
data2 = pd.read_csv(dest)
data2['Tags'] = ""
Info=[]
tagPid=defaultdict(list)

#Selecting appropriate metadata
for i in range(0,len(data.index)):
    user_tag=str(data[10][i])
    user_tag+=str(data[8][i])
    user_tag+=str(data[9][i])
    filtered = []
    filtered = list(set(filtered + tokenizer.tokenize(user_tag)))


    # removing stop words
    stop_words = set(stopwords.words('english'))
    stop_removed =[]
    for w in filtered:
        if w not in stop_words:
            stop_removed.append(w)

    # reducing words to their root form
    stemmed=[]
    lemmatizer = WordNetLemmatizer()
    for w in stop_removed:
        stemmed.append(lemmatizer.lemmatize(w))

    # obtain only nouns
    nouns = []
    for w in stemmed:
        for word,pos in nltk.pos_tag(w):
             if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                nouns.append(w)

    #Appending words in root form to file
    k=""
    final = list(set(nouns))
    for g in final:
        k+=g
        if g in tagPid.keys():
            tagPid[g].append(data[1][i])
        else:
            tagPid[g].append(data[1][i])
        k+=" "
    Info+=final
    
    data2["Tags"][i]=k
    
Info=list(set(Info))
# change file name
data2.to_csv("C:\YFCC_Dataset\Textproc1M.csv",index=False)\
newfile= "C:\YFCC_Dataset\Textproc1M.csv"
newframe = pd.read_csv(newfile)

#Binary vector to find which all tags have been used
bv = np.zeros(shape=(len(data.index),len(Info)))
int_tags={}
for w in Info:
    int_tags[w]=0
tags=newframe.Tags
for s in tags:
    i=0
    l=s.split(" ")
    count=0
    for w in Info:
        if w in l:
            count+=1
            bv[i][Info.index(w)]=1
            
            int_tags[w]+=1
            i+=1
            
#Filtering tags which appear more than 1 time
i=0
count=0
for i in int_tags.keys():
    if int_tags[i] > 1:
        count+=1
#Finding the interesting words based on threshold
interested={}
threshold = 1/len(newframe)
for w in Info:

    if int_tags[w]/len(newframe) > threshold:
        interested[w] = int_tags[w]/len(Info)
#Finding photo ID corresponding to interesting words
photo_id=[]
for w in interested.keys():
    photo_id=list(set(photo_id+tagPid[w]))
df=newframe.query('PhotoID in @photo_id')     
df.to_csv("C:\YFCC_Dataset\TP_op1M.csv",columns=['PhotoID','UserID','Latitude','Longitude','Accuracy'],index=False,header=['PhotoID','UserID','Latitude','Longitude','Accuracy'])
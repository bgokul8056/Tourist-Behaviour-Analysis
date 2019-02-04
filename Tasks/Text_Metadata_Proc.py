
# coding: utf-8

# In[1]:

import csv
import pandas as pd
location="C:\YFCC_Dataset\Filtered1M.csv"
data = pd.read_csv(location, header=None)
data.to_csv("C:\YFCC_Dataset\TP1M.csv",columns=[0,1,3,5,12,13,14],index=False,header=['LineNo','PhotoID','UserID','DateTaken','Latitude','Longitude','Accuracy'])
data.head()


# In[2]:

import csv
import nltk
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize,RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
# change file name
location="C:\YFCC_Dataset\Filtered1M.csv"
dest = "C:\YFCC_Dataset\TP1M.csv"
data = pd.read_csv(location, header=None)
tokenizer = RegexpTokenizer(r'[a-zA-Z_]+')
data2 = pd.read_csv(dest)
data2['Tags'] = ""
Info=[]
tagPid=defaultdict(list)
for i in range(0,len(data.index)):
    user_tag=str(data[10][i])
    user_tag+=str(data[8][i])
    user_tag+=str(data[9][i])
    filtered = []
    
    filtered = list(set(filtered + tokenizer.tokenize(user_tag)))

    #print(filtered)

    # removing stop words
    stop_words = set(stopwords.words('english'))
    stop_removed =[]
    for w in filtered:
        if w not in stop_words:
            stop_removed.append(w)
    #print (stop_removed)

    # reducing words to their stem form
    stemmed=[]
    lemmatizer = WordNetLemmatizer()
    for w in stop_removed:
        stemmed.append(lemmatizer.lemmatize(w))
    #print (stemmed)

    # obtain only nouns
    nouns = []
    for w in stemmed:
        for word,pos in nltk.pos_tag(w):
             if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                nouns.append(w)

    #print (nouns)
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
data2.to_csv("C:\YFCC_Dataset\Textproc1M.csv",index=False)


# In[3]:

#data3=pd.read_csv("C:\YFCC_Dataset\Textprocv2.csv")
tagPid


# In[4]:

len(Info)


# In[5]:

# change file name
newfile= "C:\YFCC_Dataset\Textproc1M.csv"
import numpy as np


# In[6]:

newframe = pd.read_csv(newfile)


# In[7]:

bv = np.zeros(shape=(len(data.index),len(Info)))
bv.shape


# In[8]:

int_tags={}
for w in Info:
    int_tags[w]=0


# In[9]:

tags=newframe.Tags
for s in tags:
    i=0
    l=s.split(" ")
    #print(l)
    count=0
    for w in Info:
        if w in l:
            count+=1
            bv[i][Info.index(w)]=1
            #print(w)
            int_tags[w]+=1
            i+=1
            
    #print(count)
    
i=0
        


# In[10]:

bv
import os


# In[11]:

# int_tags={}
# for w in Info:
#     int_tags[w]=0
#     for i in range(len(newframe.index)):
#         if bv[i][Info.index(w)]==1:
#             int_tags[w]+=1
int_tags


# In[12]:

count=0
for i in int_tags.keys():
    if int_tags[i] > 1:
        count+=1
        #print(i)
    
print(count)


# In[14]:

# int_tags['nan']


# In[15]:

1/len(newframe)


# In[16]:

interested={}
threshold = 0.0003745318352059925
for w in Info:
#     interested[w]=0.0
#     for i in range(len(newframe.index)):
    if int_tags[w]/len(newframe) > threshold:
        interested[w] = int_tags[w]/len(Info)
        


# In[17]:

len(interested)


# In[18]:

interested    


# In[19]:

photo_id=[]
for w in interested.keys():
    photo_id=list(set(photo_id+tagPid[w]))
df=newframe.query('PhotoID in @photo_id')     
df.to_csv("C:\YFCC_Dataset\TP_op1M.csv",columns=['PhotoID','UserID','Latitude','Longitude','Accuracy'],index=False,header=['PhotoID','UserID','Latitude','Longitude','Accuracy'])


# In[20]:

# df=pd.read_csv("C:\YFCC_Dataset\TP_opv2.csv",header=None)
# print(len(df))
# df.drop_duplicates(keep='first',subset=[0],inplace=True) 
# df.to_csv("C:\YFCC_Dataset\TP_opv3.csv",columns=[0,1,2,3,4],index=False,header=False)
# print(len(df))


# In[ ]:


                
                


# In[ ]:




# In[ ]:




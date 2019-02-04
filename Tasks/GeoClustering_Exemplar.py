
# coding: utf-8

# In[1]:

import math as m
import numpy as np
import pandas as pd
#change filename
uk = "C:\YFCC_Dataset\TP_op1M.csv"
rad="C:\YFCC_Dataset\Rad.csv"


# In[2]:

df = pd.read_csv(uk)
df1=df
coord={}
df.head()
#Input file


# In[3]:

len(df)


# In[4]:

Lat=np.asarray(df['Latitude'])
Long=np.asarray(df['Longitude'])
import matplotlib.pyplot as plt


# In[5]:

#creating the dataframe with coords


# In[6]:

coords = df.as_matrix(columns=['Longitude',"Latitude"])


# In[ ]:




# In[7]:

from sklearn.datasets import make_blobs
import hdbscan


# In[8]:

pd.DataFrame(coords).head()
#viewing dataframe to be passed to hdscan


# In[9]:

rads = np.radians(coords)
#converting to radians


# In[10]:

#
clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0,metric='haversine', gen_min_span_tree=True, min_cluster_size=20, min_samples=None, p=None).fit(rads)
cluster_labels = clusterer.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(-1, num_clusters)])


# In[11]:

cluster_label=(clusterer.labels_)


# In[12]:

noc=clusterer.labels_.max()
#noc - no of clusters
print(noc)


# In[13]:

# clusterer.probabilities_


# In[14]:

x = cluster_label
print(cluster_label)


# In[15]:

cluster_points={}
for i in range(0,noc+1):
    cluster_points[i]=[]
    
cluster_points[-1]=[]

for i in range(0,len(x)):
    l=cluster_points[x[i]]
    l.append(coords[i])
    cluster_points[x[i]]=l
cluster_points


# In[16]:

# No of outliers
len(cluster_points[-1])


# In[17]:

import folium
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


# In[18]:

plt.figure(figsize = (12, 12))


m = Basemap(projection='merc', resolution='l', epsg = 4269, 
            llcrnrlon=-122.567288,llcrnrlat=37.696627, urcrnrlon=-122.329308,urcrnrlat=37.852144)

# plot the aftershock
x, y = m(coords[:, 1], coords[:, 0])
m.scatter(x,y,5,marker='o',color='b')
m.arcgisimage(service='World_Shaded_Relief', xpixels = 5000, verbose= False)
plt.show()


# In[19]:

import matplotlib.cm as cmx
import matplotlib.colors as colors

# define a helper function to get the colors for different clusters
def get_cmap(N):
    '''
    Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.
    '''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='nipy_spectral') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


# In[20]:

plt.figure(figsize = (12, 12))

m = Basemap(projection='merc', resolution='l', epsg = 4269, 
            llcrnrlon=-122.567288,llcrnrlat=37.696627, urcrnrlon=-122.329308,urcrnrlat=37.852144)

unique_label = np.unique(cluster_labels)

# get different color for different cluster
cmaps = get_cmap(num_clusters)

# plot different clusters on map, note that the black dots are 
# outliers that not belone to any cluster. 
for i, cluster in enumerate(clusters):
    lons_select = cluster[:, 1]
    lats_select = cluster[:, 0]
    x, y = m(lons_select, lats_select)
    m.scatter(x,y,5,marker='o',color=cmaps(i), zorder = 10)

m.arcgisimage(service='World_Shaded_Relief', xpixels = 5000, verbose= False)

plt.show()


# In[21]:

l=list(cluster_points)
y


# In[ ]:




# In[22]:

# clusterer.minimum_spanning_tree_.plot()


# In[23]:

# clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)


# In[24]:

# clusterer.condensed_tree_.plot()


# In[ ]:




# In[25]:

# cluster_points={}
# for i in range(0,noc+1):
#     cluster_points[i]=[]
    
# cluster_points[-1]=[]

# for i in range(0,len(x)):
#     l=cluster_points[x[i]]
#     l.append(coords[i])
#     cluster_points[x[i]]=l
# cluster_points
cluster_exemplar_photoid={}
cluster_exemplar={}
for i in range(0,noc+1):
    cluster_exemplar[i]=[]
    cluster_exemplar_photoid[i]=[]
    
cluster_exemplar[-1]=[]
cluster_exemplar_photoid[-1]=[]

for i in range(0,noc):
    l=np.rad2deg(clusterer.exemplars_[i])
    cluster_exemplar[i]=l
    
        
    
    


    


# In[26]:

for i in range(0,noc):
    print(len(cluster_exemplar[i]))


# In[27]:

cluster_exemplar


# In[ ]:




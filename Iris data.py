#!/usr/bin/env python
# coding: utf-8

# In[46]:


#importing the packages
import pandas as pd
import matplotlib.pylab as plt 
import numpy as np


# In[7]:


#importing the dataset
iris1=pd.read_csv("Iris.csv")


# In[8]:


#viewing the data set 
iris1


# In[9]:


iris1.info()


# In[10]:


#dropping the species column
iris=iris1.drop(['Species','Id'],axis=1)


# In[11]:


iris.head()


# In[12]:


#normalizing function to normalize the data set 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)


# In[15]:


#normalized data frame
iris_norm=norm_func(iris.iloc[: ,:])


# In[16]:


iris_norm


# In[36]:


from sklearn.cluster import KMeans
TWSS = []
k = list(range(1, 11))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(iris_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS


# In[38]:


plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")
plt.title('The elbow method')


# In[49]:


# Applying kmeans to the dataset / Creating the kmeans classifier
x=np.array(iris_norm)
kmeans = KMeans(n_clusters = 5, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
x


# In[50]:


# Visualising the clusters - On the first two columns
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:





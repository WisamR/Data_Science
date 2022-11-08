#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. Load the iris dataset
import numpy as np
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"  
col = ["Attribute_1", "Attribute_2", "Attribute_3", "Attribute_4", "Names"]  
iris = pd.read_csv(url, names = col)

print (iris.head())


# In[2]:


# 2. Calculate "MEAN", "MEDIAN", "SD" for all attributes of iris setosa
df = pd.DataFrame(iris)
df1 = df[0:50]      # Cutting all other types of iris
print (df1.head())
print   ( "Mean for all Attribute of iris setosa ")
print (df1.mean())
print ("***************************************************")
print   ( "Median for all Attribute of iris setosa  ")
print (df1.median())
print ("***************************************************")
print ( "SD for all Attribute of iris setosa  ")
print (df1.std())
# Another way just to describe everything in iris setosa
print (" the summary of iris setosa, Attribute_2 :",df1["Attribute_2"].describe())


# In[19]:


# 3. Select one attribute (Attribute_2), plot four histograms with four methods
import matplotlib.pyplot as plt
import math as ma
from scipy.stats import iqr
kStugres =ma.log(50,2) + 1               #Sturges method
x = df1["Attribute_2"]
plt.hist(x, bins=int(kStugres))           # Histogram with "kSturges bin"
plt.title("Histogram with Sturges bins")
plt.show()
#*********************************************************************************
print ("***************************************************")

kSquare = ma.sqrt(50)                          # square root method
x = df1["Attribute_2"]
plt.hist(x, bins=int(kSquare))           # Histogram with "kSquare bin"
plt.title("Histogram with Square root bins")
plt.show()
#*********************************************************************************
print ("***************************************************")

u = iqr(df1["Attribute_2"])                    # Freedman-Diaconis
h = 2*u/(50)**(1/3)                            # the size of the bin (height) in Freedman-Diaconis
print ("The size of the bin in Freedman method", int(h), "Then number of bins is useless")
# From the boxplot for Attribute_2, we can see that the range is about 1, 
# then the numbmer of bins 1/h , then the number of bins = 1, "useless"
#*********************************************************************************
print ("*******************************************************************************")

hScot = (3.5 * df1["Attribute_2"].std())/50**(1/3)  #Scott's Rule
print ("The size of the bin in Scott Method", int(hScot), "Then number of bins is useless")
# The size of bin is 1, then the number of bin = 1, which is useless
#*********************************************************************************
# The bet with Sturges method,  n = 50
#*********************************************************************************
print (" QUESTION 3 THE SECOND PART *******************************************************************************")
# 3.Plot histograms for other iris species
print ("Histogram for iris versicolor for the same Attribute_2")
df2 = df[50:100]
print ("The table for iris versicolor ")
print (df2.head())
x = df2["Attribute_2"]
plt.hist(x, bins=5)           # Histogram with 5 bin
plt.title("Histogram for iris versicolor for the same Attribute_2 with 5 bins")
plt.show()
print ("Histogram for iris virginica for the same Attribute_2")
df3 = df[100:150]
print ("The table for iris virginica ")
print(df3.head())
x = df3["Attribute_2"]
plt.hist(x, bins=5)           # Histogram with 5 bin
plt.title("Histogram for iris virginica for the same Attribute_2 with 5 bins")
plt.show()





# In[4]:


plt.hist


# In[5]:


# 4.Boxplot plots of attributes and outliers of iris setosa?
plt.figure()
plt.boxplot(df1["Attribute_1"], 0, 'rs', 0)
print ("MAX OF ATTRIBUTE_1 ",(df1["Attribute_1"]).max())
print ("MIN OF ATTRIBUTE_1 ",(df1["Attribute_1"]).min())


# In[6]:


plt.figure()
plt.boxplot(df1["Attribute_2"], 0, 'rs', 0)
print ("MAX OF ATTRIBUTE_2 ",(df1["Attribute_2"]).max())
print ("MIN OF ATTRIBUTE_2 ",(df1["Attribute_2"]).min())


# In[7]:


plt.figure()
plt.boxplot(df1["Attribute_3"], 0, 'rs', 0)
print ("MAX OF ATTRIBUTE_3 ",(df1["Attribute_3"]).max())
print ("MIN OF ATTRIBUTE_3 ",(df1["Attribute_3"]).min())
print ("Clear outlier at 1.0, 1.7 and 1.9")


# In[8]:


plt.figure()
plt.boxplot(df1["Attribute_4"], 0, 'rs', 0)
print ("MAX OF ATTRIBUTE_4 ",(df1["Attribute_4"]).max())
print ("MIN OF ATTRIBUTE_4 ",(df1["Attribute_4"]).min())
print ("Clear outlier at 0.5 and 0.6")


# In[20]:


# 4. Calculation of Pearson’s correlation tables
from scipy.stats import pearsonr

print ("***********************************************************")
print ("Pearson’s correlation tables")
print (df[["Attribute_1", "Attribute_2"]].corr())
print (df[["Attribute_1", "Attribute_3"]].corr())
print (df[["Attribute_1", "Attribute_4"]].corr())
print (df[["Attribute_2", "Attribute_3"]].corr())
print (df[["Attribute_2", "Attribute_4"]].corr())
print (df[["Attribute_3", "Attribute_4"]].corr())




# In[10]:


from scipy.stats import spearmanr
rank_cor12=spearmanr(df["Attribute_1"], df["Attribute_2"])
print (rank_cor12)

rank_cor13=spearmanr(df["Attribute_1"], df["Attribute_3"])
print (rank_cor13)

rank_cor14=spearmanr(df["Attribute_1"], df["Attribute_4"])
print (rank_cor14)

rank_cor23=spearmanr(df["Attribute_2"], df["Attribute_3"])
print (rank_cor23)

rank_cor24=spearmanr(df["Attribute_2"], df["Attribute_4"])
print (rank_cor24)

rank_cor34=spearmanr(df["Attribute_3"], df["Attribute_4"])
print (rank_cor34)


# In[11]:


from scipy.stats import kendalltau
print (kendalltau(df["Attribute_1"], df["Attribute_2"]))
print(kendalltau(df["Attribute_1"], df["Attribute_3"]))
print(kendalltau(df["Attribute_1"], df["Attribute_4"]))
print(kendalltau(df["Attribute_2"], df["Attribute_3"]))
print(kendalltau(df["Attribute_2"], df["Attribute_4"]))
print(kendalltau(df["Attribute_3"], df["Attribute_4"]))


# In[12]:


# The best correlation happened as bellow
plt.scatter(df["Attribute_3"], df["Attribute_4"])


# In[13]:


# The best correlation happened as bellow
plt.scatter(df["Attribute_1"], df["Attribute_4"])


# In[14]:


# The best correlation happened as bellow
plt.scatter(df["Attribute_1"], df["Attribute_3"])


# In[15]:


# The Worst correlation happened as bellow
plt.scatter(df["Attribute_2"], df["Attribute_1"])


# In[16]:


#6. Principal component analysis (PCA) with and without z-score normalization: project the data
#to the first two principal components. Visualize the result as a scatter plot. What is the
#proportion of variance explained in projections? What can be observed?


# In[ ]:





# In[22]:


# 6. Principal component analysis (PCA) with z-score normalization (Standarization)
from sklearn.preprocessing import StandardScaler
features = ["Attribute_1", "Attribute_2", "Attribute_3", "Attribute_4"]
# Separating out the features
x1 = df.loc[:, features].values
# Separating out the target
y = df.loc[:,["Names"]].values
# Standardizing the features
x = StandardScaler().fit_transform(x1)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['Names']]], axis = 1)

fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
Names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
colors = ['r', 'g', 'b']
for Names, color in zip(Names, colors):
    indicesToKeep = finalDf['Names'] == Names
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 25)
ax.legend(["Iris-setosa", "Iris-versicolor", "Iris-virginica"], loc = "lower right" ,fontsize =10, borderpad=3)
ax.grid()


# In[ ]:





# In[21]:


# Principal component analysis (PCA) without z-score normalization
principalComponents = pca.fit_transform(x1)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['Names']]], axis = 1)

fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
Names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
colors = ['r', 'g', 'b']
for Names, color in zip(Names, colors):
    indicesToKeep = finalDf['Names'] == Names
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 25)
ax.legend(["Iris-setosa", "Iris-versicolor", "Iris-virginica"], loc = "lower right" ,fontsize =10, borderpad=3)
ax.grid()


# In[ ]:





# In[ ]:





# In[ ]:





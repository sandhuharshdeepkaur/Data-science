# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:09:48 2021

@author: sandh
"""
#importing useful libraries used in case segmentation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np




# Reading csv(i.e. A comma Separated Values) file into dataframe
df = pd.read_csv('C:/Users/sandh/OneDrive/Desktop/Mall_Customers.csv')
df.head(200)

# Determining shape, type and describing the dataframe
df.shape

df.dtypes

df.describe()


# Visulizing Gender
plt.figure(figsize=(3,4))
sns.countplot(x=df.Gender)
plt.show()
# Plot pairwaise relationships in Datasets 
sns.pairplot(df, vars =["Age", "Annual Income (k$)", "Spending Score (1-100)"], hue='Gender',kind = 'reg')

# Plot rectangular data as a color-encoded matrix.
df_corr = df.corr(method='spearman')
plt.figure(figsize=(8,8))
sns.heatmap(df_corr)

# Visulizing Age
age_1 = df.Age[(df.Age >= 18) & (df.Age <= 25)]
age_2 = df.Age[(df.Age >= 26) & (df.Age <= 35)]
age_3 = df.Age[(df.Age >= 36) & (df.Age <= 45)]
age_4 = df.Age[(df.Age >= 46) & (df.Age <= 55)]
age_5 = df.Age[df.Age >= 56]
agex = ["18-25", "26-35", "36-45", "46-55", "55+"]
agey = [len(age_1.values),len(age_2.values),len(age_3.values),len(age_4.values),len(age_5.values)]
plt.figure(figsize= (3,4))
sns.barplot(x = agex, y = agey)
plt.title("No. of Customers and Ages")
plt.xlabel('Age')
plt.ylabel('No. of Customers')
plt.show()
                                                                                                              
# Specifying number of clusters for Age, Annual Income and Spending Score with K-Means clustering
km = KMeans(n_clusters=5)
km
# Computing cluster centers and predicting cluster index for each sample in an array
y_predicted = km.fit_predict(df[['Annual Income (k$)', 'Spending Score (1-100)']]) 
y_predicted

# Finding cluster in the dataframe
df['cluster'] = y_predicted
df.head(200)

# Determinig centroids
km.cluster_centers_

# The Elbow Point Graph
k_rng= range(1,10)
sse = []

for k in k_rng:
  kmeans = KMeans(n_clusters=k)
  kmeans.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])

  sse.append(kmeans.inertia_)
sse

plt.plot(k_rng,sse,('bx-'))
sns.set()
plt.plot(k_rng, sse)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()
# Analyzing Annual Income and Spending score 
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
df4 = df[df.cluster == 3]
df5 = df[df.cluster == 4]
plt.scatter(df1['Annual Income (k$)'], df1['Spending Score (1-100)'], s=20, color = 'green')
plt.scatter(df2['Annual Income (k$)'], df2['Spending Score (1-100)'], s=20, color = 'red')
plt.scatter(df3['Annual Income (k$)'], df3['Spending Score (1-100)'], s=20, color = 'black')
plt.scatter(df4['Annual Income (k$)'], df4['Spending Score (1-100)'], s=20, color = 'blue')
plt.scatter(df5['Annual Income (k$)'], df5['Spending Score (1-100)'], s=20, color = 'yellow')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=100, c='cyan', marker = '^', label='Centroids')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

# Analyzing the Age, Annual Income and the Spending Score with KMeans Clustering
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
th = fig.add_subplot(111, projection='3d')
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
df4 = df[df.cluster == 3]
df5 = df[df.cluster == 4]
th.scatter(df1['Age'], df1['Annual Income (k$)'], df1['Spending Score (1-100)'], s=20, color = 'green')
th.scatter(df2['Age'], df2['Annual Income (k$)'], df2['Spending Score (1-100)'], s=20, color = 'red')
th.scatter(df3['Age'], df3['Annual Income (k$)'], df3['Spending Score (1-100)'], s=20, color = 'black')
th.scatter(df4['Age'], df4['Annual Income (k$)'], df4['Spending Score (1-100)'], s=20, color = 'blue')
th.scatter(df5['Age'], df5['Annual Income (k$)'], df5['Spending Score (1-100)'], s=20, color = 'yellow')
th.view_init()
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
th.set_zlabel('Spending Score (1-100)')
plt.show()

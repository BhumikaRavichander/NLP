#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist

df = pd.read_json(r'DataP7.txt', lines=True)
df.head()

df.title.head()

vectorizer = TfidfVectorizer(max_features=2**12)
X = vectorizer.fit_transform(df['title'].values)

pca = PCA(n_components=3)
X_embedded = pca.fit_transform(X.toarray())

distortions = []
K = range(2, 50)
for k in K:
    k_means = KMeans(n_clusters=k, random_state=42).fit(X_embedded)
    k_means.fit(X_embedded)
    distortions.append(sum(np.min(cdist(X_embedded, k_means.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

X_line = [K[0], K[-1]]
Y_line = [distortions[0], distortions[-1]]

k = 20
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X_embedded)
y = y_pred

df['cluster'] = pd.Series(y, index=df.index)

gk = df.groupby('cluster') 

for i in range(0,15):
    print(gk.get_group(i)


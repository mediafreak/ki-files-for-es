"""
============================
A demo of K-Means clustering 
============================

"""
print(__doc__)

from time import time
import numpy as np
import pylab as pl
import sys

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed() # an integer (e.g. 42) FIXES the seed!

# load data
data = np.loadtxt('spaeth_01.txt')         # training data

n_samples, n_features = data.shape

# specify parameters of kmeans
max_iter  = 100
initialization = "random" # "manual", "k-means++", "random" or "pca"

if initialization == "manual":
    init_cluster = np.array([ [10.0,35.0], [35.0,15.0],  [40.0,40.0] ])
    n_cluster = init_cluster.shape[0]
    # number of cluster determined from size of init_cluster
    # same for pca initialization

elif initialization == "k-means++" or initialization == "random":
    n_cluster = 3

elif initialization == "pca":
    print("pca initialization currently not implemented!")
    sys.exit()

print("number of features: %d" % n_features)
print("number of samples: %d" % n_samples)
print("number of cluster: %d" % n_cluster)
print()


# kmeans initialization
if initialization == "manual":
    print("manual initialization with")
    print(init_cluster)
    kmeans = KMeans(init=init_cluster, n_clusters=n_cluster, n_init=1, max_iter=max_iter).fit(data)

elif initialization == "k-means++":
    print("k-means++ initialization")
    kmeans = KMeans(init='k-means++', n_clusters=n_cluster, n_init=1, max_iter=max_iter).fit(data)
    
elif initialization == "random":
    print("random initialization")    
    kmeans = KMeans(init='random', n_clusters=n_cluster, n_init=1, max_iter=max_iter).fit(data)

elif initialization == "pca":
    print("pca initialization")
    # PCA components initialization
    pca = PCA(n_components=n_cluster).fit(data)
    kmeans = KMeans(init=pca.components_, n_clusters=n_cluster, n_init=1).fit(data)

else:
    print("unknown initialization")
  
print()  
print("cluster center: ")
print(kmeans.cluster_centers_)

# For plotting of the decision boundaries: define mesh
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
pl.set_cmap(pl.cm.Paired)
pl.pcolormesh(xx, yy, Z)

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
pl.scatter(centroids[:, 0], centroids[:, 1],
           marker='x', s=169, linewidths=3,
           color='w')
           
#plot also the training points
pl.scatter(data[:,0], data[:,1], edgecolor='black', facecolor='none', marker='o', s=20)
           
           
pl.title('K-means clustering algorithm\n'
         'Final cluster center are marked with white cross')
pl.axis('tight')
pl.show()

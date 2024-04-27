#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
# import necessary libraries and specify that graphs should be plotted inline.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

iris = datasets.load_iris()
X = iris.data[:, :2]  
y = iris.target


# In[10]:


n_neighbors = 3
clf = neighbors.KNeighborsClassifier(n_neighbors)
clf = clf.fit(X, y)


# In[17]:


print('The k nearest neighbors (and the corresponding distances) to user [1, 1] are:', clf.kneighbors([[1., 1.]]))

print('The k nearest neighbors to each user are:', clf.kneighbors(X, return_distance=False) )

A = clf.kneighbors_graph(X)
A.toarray()


# In[19]:


## Visualization of the decision boundaries

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

n_neighbors = 15
weights = 'distance'
#for weights in ['uniform', ]: 
for n_neighbors in [5,10]:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    # For supervised learning applications, this accepts two arguments: the data X and the labels y (e.g. model.fit(X, y))
    clf.fit(X, y) # we train again the model as we will use only two variables to visualize the decision boundaries

    # Plot the decision boundary. 
    # For that, we will assign a color to each point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    #plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()


# In[27]:


## Nearest Neighbors regression

# This following example demonstrates the resolution of a regression problem using a k-Nearest Neighbor 
# and the interpolation of the target using both barycenter and constant weights.

# Generate sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))


# In[28]:


# Fit regression model
n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, c='k', label='data')
    plt.plot(T, y_, c='g', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights))

plt.show()


# In[ ]:





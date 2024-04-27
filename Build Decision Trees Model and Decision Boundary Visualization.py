#!/usr/bin/env python
# coding: utf-8

# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import numpy as np


# In[41]:


iris = load_iris()
#print(iris.DESCR)


# In[44]:


clf = tree.DecisionTreeClassifier(max_depth =6)
clf = clf.fit(iris.data, iris.target)
#tree.plot_tree(clf)
scores = cross_val_score(clf, iris.data, iris.target, cv=10)


# In[46]:


print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print('The probabilities of belonging to each one of the classes are estimated as:', clf.predict_proba(iris.data[:1, :]))
# Here you can see the probability of each one of the classes


# In[47]:


# Decision Boundary Visualization:
# Plot the decision surface of a decision tree

# Parameters
featureA, featureB = 0, 3 # select two variables to visualize
plot_colors = "bry"
plot_step = 0.02  # step size in the mesh

X = iris.data[:, [featureA, featureB] ] # We only take the two corresponding features
y = iris.target

n_classes = len(set(y))

# Train
clf = tree.DecisionTreeClassifier().fit(X, y)

# Plot the decision boundary. 
# For that, we will assign a color to each point in the mesh [x_min, m_max]x[y_min, y_max].   
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

plt.xlabel(iris.feature_names[featureA])
plt.ylabel(iris.feature_names[featureB])
plt.axis("tight")

# Plot the training points
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                cmap=plt.cm.Paired)

plt.axis("tight")

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend()
plt.show()


# In[ ]:





# Clustering

>* It is basically a type of unsupervised learning method . An unsupervised learning method is a method in which we draw references from datasets consisting of input data 
without labelled responses. Generally, it is used as a process to find meaningful structure, explanatory underlying processes, generative features, and groupings inherent 
in a set of examples.

>* Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data 
points in the same group and dissimilar to the data points in other groups. It is basically a collection of objects on the basis of similarity and dissimilarity between them.


* Clustering can be considered the most important unsupervised learning problem; so, as every other problem of this kind,
it deals with finding a structure in a collection of unlabeled data.
* A loose definition of clustering could be “the process of organizing objects into groups whose members are similar in some way”.
* A cluster is therefore a collection of objects which are “similar” between them and are “dissimilar” to the objects belonging to other clusters.

### Why Clustering ?
Clustering is very much important as it determines the intrinsic grouping among the unlabeled data present. There are no criteria for a good clustering. 
It depends on the user, what is the criteria they may use which satisfy their need. For instance, we could be interested in finding representatives for homogeneous groups
(data reduction), in finding “natural clusters” and describe their unknown properties (“natural” data types), in finding useful and suitable groupings (“useful” data classes)
or in finding unusual data objects (outlier detection). This algorithm must make some assumptions which constitute the similarity of points and each assumption make different 
and equally valid clusters.



### Clustering Methods :

>* Density-Based Methods : These methods consider the clusters as the dense region having some similarity and different from the lower dense region of the space. These methods have good accuracy and ability to merge two clusters.Example DBSCAN (Density-Based Spatial Clustering of Applications with Noise) , OPTICS (Ordering Points to Identify Clustering Structure) etc.
>* Hierarchical Based Methods : The clusters formed in this method forms a tree-type structure based on the hierarchy. New clusters are formed using the previously formed one. It is divided into two category
* Agglomerative (bottom up approach)
* Divisive (top down approach)
examples CURE (Clustering Using Representatives), BIRCH (Balanced Iterative Reducing Clustering and using Hierarchies) etc.

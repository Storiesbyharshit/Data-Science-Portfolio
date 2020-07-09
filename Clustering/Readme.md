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

* **Density-Based Methods** : These methods consider the clusters as the dense region having some similarity and different from the lower dense region of the space. These methods have good accuracy and ability to merge two clusters.Example DBSCAN (Density-Based Spatial Clustering of Applications with Noise) , OPTICS (Ordering Points to Identify Clustering Structure) etc.
* **Hierarchical Based Methods** : The clusters formed in this method forms a tree-type structure based on the hierarchy. New clusters are formed using the previously formed one. It is divided into two category 
  - Agglomerative (bottom up approach)
  - Divisive (top down approach)
examples CURE (Clustering Using Representatives), BIRCH (Balanced Iterative Reducing Clustering and using Hierarchies) etc.




### Applications of Clustering
Clustering has a large no. of applications spread across various domains. Some of the most popular applications of clustering are:

* Recommendation engines
* Market segmentation
* Social network analysis
* Search result grouping
* Medical imaging
* Image segmentation
* Anomaly detection

#### Clustering Algorithms

Several approaches to clustering exist.Below is a short discussion of four common approaches :
#### Centroid-based Clustering
>Centroid-based clustering organizes the data into non-hierarchical clusters, in contrast to hierarchical clustering defined below. k-means is the most widely-used centroid-based clustering algorithm. Centroid-based algorithms are efficient but sensitive to initial conditions and outliers. This course focuses on k-means because it is an efficient, effective, and simple clustering algorithm.


#### Density-based Clustering
>Density-based clustering connects areas of high example density into clusters. This allows for arbitrary-shaped distributions as long as dense areas can be connected. These algorithms have difficulty with data of varying densities and high dimensions. Further, by design, these algorithms do not assign outliers to clusters.


#### Distribution-based Clustering
>This clustering approach assumes data is composed of distributions, such as Gaussian distributions. In Figure 3, the distribution-based algorithm clusters data into three Gaussian distributions. As distance from the distribution's center increases, the probability that a point belongs to the distribution decreases. The bands show that decrease in probability. When you do not know the type of distribution in your data, you should use a different algorithm.


#### Hierarchical Clustering
>Hierarchical clustering creates a tree of clusters. Hierarchical clustering, not surprisingly, is well suited to hierarchical data, such as taxonomies. See Comparison of 61 Sequenced Escherichia coli Genomes by Oksana Lukjancenko, Trudy Wassenaar & Dave Ussery for an example. In addition, another advantage is that any number of clusters can be chosen by cutting the tree at the right level.



<p align="center">
  <img alt="For-the-Badge-Python" src="http://ForTheBadge.com/images/badges/made-with-python.svg">
  
</p>


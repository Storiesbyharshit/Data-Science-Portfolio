# Principal Component Analysis

Given a collection of points in two, three, or higher dimensional space, a "best fitting" line can be defined as one that minimizes
the average squared distance from a point to the line. The next best-fitting line can be similarly chosen from directions perpendicular to the first.
Repeating this process yields an orthogonal basis in which different individual dimensions of the data are uncorrelated.
These basis vectors are called principal components, and several related procedures principal component analysis (PCA).

The main idea of principal component analysis (PCA) is to reduce the dimensionality of a data set consisting of many variables correlated with each other, 
either heavily or lightly, while retaining the variation present in the dataset, up to the maximum extent. The same is done by transforming the variables 
to a new set of variables, which are known as the principal components (or simply, the PCs) and are orthogonal, ordered such that the retention of variation present 
in the original variables decreases as we move down in the order. So, in this way, the 1st principal component retains maximum variation that was present in the original 
components. The principal components are the eigenvectors of a covariance matrix, and hence they are orthogonal.

Principal component analysis helps make data easier to explore and visualize. It is a simple non-parametric technique for extracting 
information from complex and confusing data sets. Principal component analysis is focused on the maximum variance amount with the fewest number of principal components.
One of the distinct advantages associated with the principal component analysis is that once patterns are found in the concerned data, compression of data is also supported.
One makes use of principal component analysis to eliminate the number of variables or when there are too many predictors compared to number of observations or
to avoid multicollinearity.



<p align="center">
  <img alt="For-the-Badge-Python" src="http://i.stack.imgur.com/gZMOV.png">
  
</p>


### Content :

*  [Removing-Multicollinearity-using-PCA](http://github.com/Storiesbyharshit/Data-Science-Portfolio/tree/master/PCA/Removing-Multicollinearity-using-PCA)
> Removing the existing multicollinearity in the dataset using PCA.
*  [Wine-Classification-PCA](http://github.com/Storiesbyharshit/Data-Science-Portfolio/tree/master/PCA/Wine-Classification-PCA)
> Reducing features of the dataset to principal components and then performing and visualising decision boundries.


<p align="center">
  <img alt="For-the-Badge-Python" src="http://ForTheBadge.com/images/badges/made-with-python.svg">
  
</p>

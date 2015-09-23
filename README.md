# Clustering
Goal of this project is to implement K-means clustering and GMM clustering without any Machine Learning libraries

# K-means Clustering

Implemented k-means clustering on six various data sets using Sum Sqaured Error criteria.
In order to have a suitable clustering, I tried to find the number of clusters (k) that produce the best clustering.
For k-Means this can be achieved by trying different values of k and tracking the SSE criterion.

#  Gaussian Mixture Models Clustering (GMM) 

I implemented GMM clustering algorithm to cluster all six datasets. In order to have a suitable clustering
I tried to find the number of clusters (k) that produce the best clustering using Sum Squared Error and 
Normalized Mutual Information (NMI) criteria. 

Data sets for this project:

1. Dermatology: 366 instances, 34 features and 6 classes:   https://archive.ics.uci.edu/ml/datasets/Dermatology
2. Vowels: 990 instances, 10 features and 11 classes : https://archive.ics.uci.edu/ml/datasets/Japanese+Vowels
3. Glass: 214 instances, 9 features and 6 classes. https://archive.ics.uci.edu/ml/datasets/Glass+Identification
4. Ecoli: 327 instances, 7 features and 5 classes. https://archive.ics.uci.edu/ml/datasets/Ecoli
5. Yeast: 1479 instances, 8 features and 9 classes. https://archive.ics.uci.edu/ml/datasets/Yeast
6. Soybean: 290 instances, 35 features and 15 classes. https://archive.ics.uci.edu/ml/datasets/Soybean+%28Large%29

 All the datasets are multi-class classification datasets i.e., have more than two classes.





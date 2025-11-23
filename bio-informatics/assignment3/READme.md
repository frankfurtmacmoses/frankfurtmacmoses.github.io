This assignment is to utilize students' knowledge in clustering analysis, dimensionality reduction, and data visualization to determine the optimal number of clusters and to investigate the effect of dimensionality reduction in such analysis.  Download the data (data_10v.csv Download data_10v.csv).  This is a comma-separated file, containing 11 columns (sample_id, V1, V2, ..., V10) and 500 samples.  The samples are generated from k multivariate normally distributed models.  The number of samples in each group varies.

[25 pts] Use average silhouette width as a metric to assess the quality of clusters to identify k.  One can use a clustering method with varying k and compute average silhouette width and pick k with the highest average silhouette width.
Use k-means clustering to identify k.
Use hierarchical clustering with average linkage to identify k.
[25 pts] Principal component analysis (PCA) is a method to reduce the dimensionality of data and retain variance (information) contained in the data.  The data with reduced dimensionality can be analyzed with lower computational complexity.
Perform Principal component analysis to obtain the data with reduced dimensionality.  This can be done using 'prcomp' function in R.  The data with reduced dimensionality can be accessed as follows:
     >> d_pca <- prcomp(df)
     >> d_pca$x   # data with reduced dimensionality
Identify the number of principal components to retain >75% of variability.  The percentage of variability explained by each principal component can be computed by the ratio between stdev_k^2 / sum(stdev_j^2).  'stdev' is another slot in the output of 'prcomp'.  If PC1 can explain 24% of the total variability, PC2 17%, etc., PC1 and PC2 together can explain 41% of the total variability.
[25 pts] Repeat #1 using the data with reduced dimensionality from #2 -- determine k using k-means and hierarchical clustering with average linkage with average silhouette width as metric.
[25 pts] Perform clustering with the optimal k identified above and assign cluster ID to each sample.  Visualize PCA plot with each sample color-coded by cluster ID.  [Hint: 'autoplot' function in ggfortify R package can be used.]
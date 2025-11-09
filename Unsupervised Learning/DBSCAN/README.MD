# DBSCAN Clustering

**Clustering** is a type of unsupervised learning used to uncover hidden structure in data. Unlike supervised learning, which requires labeled outcomes, clustering methods aim to group similar observations together without prior labels. Real-world applications include customer segmentation, anomaly detection, and grouping spatial/geographic data.

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is a powerful clustering algorithm that groups points based on their density rather than distance to a centroid. Unlike K-means, it does not require specifying the number of clusters in advance and can naturally identify clusters of arbitrary shape, as well as outliers.

At a high level, DBSCAN works by:

1. Defining two key parameters:

   * **\$\varepsilon\$ (eps):** the maximum distance to consider two points as neighbors.
   * **MinPts:** the minimum number of neighbors required to form a dense region.
2. Classifying points as:

   * **Core points:** have at least `MinPts` neighbors within \$\varepsilon\$.
   * **Border points:** fall within the neighborhood of a core point but don’t have enough neighbors to be a core themselves.
   * **Noise (outliers):** points that are not part of any cluster.
3. Expanding clusters by connecting dense regions of core points and their neighbors.

This allows DBSCAN to discover clusters of irregular shapes and to **automatically identify noise/outliers** in the dataset.

In `dbscan_clustering.ipynb`, we:

* Explore the **mathematics and intuition** behind DBSCAN
* Apply DBSCAN on synthetic datasets (e.g., two moons, circles, blobs)
* Add random noise and use DBSCAN to **identify and highlight outliers**
* Evaluate cluster quality using the **Silhouette Score**
* Visualize which outliers DBSCAN correctly identified vs. those it missed
* Compare DBSCAN’s strengths and weaknesses relative to K-means


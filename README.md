📝 Explanation of Each Step

Load and Prepare Dataset

Use a dataset like Iris, which has natural groupings.

PCA is used here to reduce dimensions for visualization, not for clustering itself.

Elbow Method for Optimal K

Try different k values.

Plot inertia (sum of squared distances to centroids).

Look for the "elbow" point where adding more clusters doesn't improve much.

Fit K-Means & Assign Cluster Labels

Choose the optimal k from the elbow curve (e.g., k=3).

Use fit_predict() to assign labels.

Visualize Clusters

Use matplotlib to color points based on their assigned cluster.

PCA helps show clusters in 2D space.

Evaluate Clustering

Use Silhouette Score (range: -1 to 1).

Closer to 1 = better separation between clusters.

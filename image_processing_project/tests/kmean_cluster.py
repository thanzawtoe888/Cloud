import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. Create some sample 2D data
# We'll create three distinct groups of points
data = np.float32(np.vstack([
    np.random.rand(50, 2) * 10,       # Group 1
    np.random.rand(50, 2) * 10 + 20,  # Group 2
    np.random.rand(50, 2) * 10 + 40   # Group 3
]))

# 2. Define the number of clusters (K)
K = 3

# 3. Define the termination criteria
# Stop if 100 iterations are reached or epsilon (accuracy) of 0.85 is achieved
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# 4. Apply K-Means clustering
# 'attempts' specifies how many times to run the algorithm with different initial centroids.
# 'cv2.KMEANS_RANDOM_CENTERS' randomly selects initial centroids.
compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 5. Visualize the results
# Plot the original data points, colored by their assigned cluster
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels.flatten(), cmap='viridis', s=50, alpha=0.7)

# Plot the cluster centers in red
plt.scatter(centers[:, 0], centers[:, 1], s=200, marker='X', c='red', edgecolor='black', label='Cluster Centers')

plt.title('K-Means Clustering Example')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.legend()
plt.grid(True)
plt.show()

print("Cluster Centers:\n", centers)
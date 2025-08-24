"""A3. Take any two feature vectors from your dataset. Calculate the Minkowski distance with r from 1 
to 10. Make a plot of the distance and observe the nature of this graph.
Minkowski distance formula:
For two vectors x and y, and order r:
    D(x, y) = (sum_i |x_i - y_i|^r)^(1/r) r=1 manhattan, r=2 euclidean 
"""

import numpy as np
import matplotlib.pyplot as plt

# Load PCA features
data = np.load(r"C:\Users\Vasanth Kumar\Desktop\features_pca.npz", allow_pickle=True)
X, y = data['Xp'], data['y']

# Pick two feature vectors: one non-venomous, one venomous
x1 = X[y == 0][0]   # first non-venomous snake
x2 = X[y == 1][0]   # first venomous snake

# Calculate Minkowski distances for r=1 to 10
distances = []
r_values = range(1, 11)

for r in r_values:
    d = np.linalg.norm(x1 - x2, ord=r)
    distances.append(d)

# Print results
print("Minkowski distances between one non-venomous and one venomous snake:")
for r, d in zip(r_values, distances):
    print(f"r={r} → distance={d:.4f}")

# Plot the distances
plt.figure(figsize=(8,5))
plt.plot(r_values, distances, marker='o', linestyle='-', color="blue")
plt.title("Minkowski Distance between Two Feature Vectors")
plt.xlabel("r (order)")
plt.ylabel("Distance")
plt.grid(alpha=0.3)
plt.show()

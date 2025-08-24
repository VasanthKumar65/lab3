"""A2. Take any feature from your dataset. Observe the density pattern for that feature by plotting the 
histogram. Use buckets (data in ranges) for histogram generation and study. Calculate the mean and 
variance from the available data.  
(Suggestion: numpy.histogram()gives the histogram data. Plot of histogram may be 
achieved with matplotlib.pyplot.hist())"""


import numpy as np
import matplotlib.pyplot as plt

# Load PCA features
data = np.load(r"C:\Users\Vasanth Kumar\Desktop\features_pca.npz", allow_pickle=True)
X, y = data['Xp'], data['y']

# Pick one feature (e.g., first PCA feature = column 0)
feature_idx = 3
feature_values = X[:, feature_idx]

# Calculate mean and variance
mean_val = np.mean(feature_values)
var_val = np.var(feature_values)

print(f"Feature {feature_idx} → Mean: {mean_val:.4f}, Variance: {var_val:.4f}")

# Plot histogram
plt.figure(figsize=(8,5))
plt.hist(feature_values, bins=30, color="skyblue", edgecolor="black")
plt.title(f"Histogram of Feature {feature_idx}")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.show()

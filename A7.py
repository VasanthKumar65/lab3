"""A7. Use the predict() function to study the prediction behavior of the classifier for test vectors. 
>>> neigh.predict(X_test) 
Perform classification for a given vector using neigh.predict(<<test_vect>>). This shall produce the 
class of the test vector (test_vect is any feature vector from your test set)."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data = np.load(r"C:\Users\Vasanth Kumar\Desktop\features_pca.npz", allow_pickle=True)
X, y = data['Xp'], data['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train kNN model (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ---- A7: Predictions ----
# Predict labels for all test samples
y_pred = knn.predict(X_test)

print("Predictions for first 10 test samples:", y_pred[:10])
print("True labels for first 10 test samples:", y_test[:10])

# Predict for a single test vector
sample_idx = 0
single_pred = knn.predict([X_test[sample_idx]])  # must be 2D input
print(f"\nPrediction for test sample {sample_idx}: {single_pred[0]}")
print(f"True label for test sample {sample_idx}: {y_test[sample_idx]}")

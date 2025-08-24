"""A6. Test the accuracy of the kNN using the test set obtained from above exercise. Following code for 
help. 
>>> neigh.score(X_test, y_test) """


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# --- Load dataset (from PCA features file) ---
data = np.load(r"C:\Users\Vasanth Kumar\Desktop\features_pca.npz", allow_pickle=True)
X, y = data['Xp'], data['y']

# --- A4: Train-test split (70% train, 30% test) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Training set size:", X_train.shape, y_train.shape)
print("Testing set size:", X_test.shape, y_test.shape)

# --- A5: Train kNN classifier (k=3) ---
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("✅ kNN model trained with k=3")

# --- A6: Test Accuracy on unseen test set ---
accuracy = knn.score(X_test, y_test) #.score() implicitly calls predict() and compares with y_test
print(f"✅ Test Accuracy of kNN (k=3): {accuracy*100:.2f}%")

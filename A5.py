"""A5. Train a kNN classifier (k =3) using the training set obtained from above exercise. Following code 
for help: 
>>> import numpy as np 
>>> from sklearn.neighbors import KNeighborsClassifier 
>>> neigh = KNeighborsClassifier(n_neighbors=3) 
>>> neigh.fit(X, y) """


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load preprocessed features
data = np.load(r"C:\Users\Vasanth Kumar\Desktop\features_pca.npz", allow_pickle=True)
X, y = data['Xp'], data['y']

# Split dataset into train/test (same as A4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Define kNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3) #default is eucledian distance

# Train (fit) the model
knn.fit(X_train, y_train)

print("✅ kNN model (k=3) trained successfully on training set!")

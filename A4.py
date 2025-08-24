"""A4. Divide dataset in your project into two parts – train & test set. To accomplish this, use the train
test_split() function available in SciKit. See below sample code for help: 
>>> import numpy as np 
>>> from sklearn.model_selection import train_test_split 
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) """

import numpy as np
from sklearn.model_selection import train_test_split

# Load PCA-preprocessed dataset
data = np.load(r"C:\Users\Vasanth Kumar\Desktop\features_pca.npz", allow_pickle=True)
X, y = data['Xp'], data['y']

# Split dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Print shapes
print("Training set size:", X_train.shape, y_train.shape)
print("Testing set size:", X_test.shape, y_test.shape)

# Check class balance in train/test sets
print("\nClass distribution in training set:", np.bincount(y_train))
print("Class distribution in testing set:", np.bincount(y_test))

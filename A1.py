"""A1. Evaluate the intraclass spread and interclass distances between the classes in your dataset. If 
your data deals with multiple classes, you can take any two classes. Steps below (refer below 
diagram for understanding): 
• Calculate the mean for each class (also called as class centroid) 
(Suggestion: You may use numpy.mean() function for finding the average vector for all 
vectors in a given class. Please define the axis property appropriately to use this function. EX: 
feat_vecs.mean(axis=0)) 
• Calculate spread (standard deviation) for each class 
(Suggestion: You may use numpy.std() function for finding the standard deviation vector 
for all vectors in a given class. Please define the axis property appropriately to use this 
function.) 
• Calculate the distance between mean vectors between classes 
(Suggestion: numpy.linalg.norm(centroid1 – centroid2) gives the Euclidean 
distance between two centroids.)"""

import numpy as np

def compute_class_stats(X, y):
    X0 = X[y == 0]
    X1 = X[y == 1]

    centroid_0 = np.mean(X0, axis=0)
    centroid_1 = np.mean(X1, axis=0)

    spread_0 = np.std(X0, axis=0)
    spread_1 = np.std(X1, axis=0)

    interclass_dist = np.linalg.norm(centroid_0 - centroid_1)

    return centroid_0, centroid_1, spread_0, spread_1, interclass_dist


if __name__ == "__main__":
    data = np.load(r"C:\Users\Vasanth Kumar\Desktop\features_pca.npz", allow_pickle=True)
    X, y = data['Xp'], data['y']

    c0, c1, s0, s1, dist = compute_class_stats(X, y)

    # Print full results
    np.set_printoptions(precision=4, suppress=True)  # nicer formatting
    print("Centroid (Class 0 - Non-venomous):\n", c0)
    print("Centroid (Class 1 - Venomous):\n", c1)
    print("Spread (Class 0):\n", s0)
    print("Spread (Class 1):\n", s1)
    print("\nInter-class distance (Euclidean):", dist)


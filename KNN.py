import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

A = np.array([[3.2, 2.1], [2.3, 4.3], [3.9, 3.6], [3.8, 6.4], [4.8, 2.0], 
             [8.4, 3.2], [5.1, 7.6], [4.9, 4.7], [3.6, 5.2], [4.5, 3.0],])

k = 5

test_data = [3.2, 2.8]

plt.figure()
plt.title('Input Data')
plt.scatter(A[:,0], A[:,1], marker = 'o', s = 100, color = 'r')

knn_model = NearestNeighbors(n_neighbors = k, algorithm = 'auto').fit(A)
distances, indices = knn_model.kneighbors([test_data])

print("\nK Nearest Neighbors:")
for rank, index in enumerate(indices[0][:k], start = 1):
   print(str(rank) + " is", A[index])

plt.figure()
plt.title('K Nearest Neighbors')
plt.scatter(A[:, 0], A[:, 1], marker = 'o', s = 100, color = 'g')
plt.scatter(A[indices][0][:][:, 0], A[indices][0][:][:, 1],
   marker = 'o', s = 250, color = 'g', facecolors = 'none')
plt.scatter(test_data[0], test_data[1],
   marker = 'x', s = 100, color = 'b')
plt.show()



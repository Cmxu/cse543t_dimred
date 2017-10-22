import numpy as np

N = 1000

X = np.concatenate((np.zeros((1, N-1)), np.eye(N-1)), axis = 0)
X = np.concatenate((X, np.asarray([i for i in range(N)])[:,None]), axis = 1)

print("Initial Space")

print(X)

cov = np.cov(X.T * X)

D,V = np.linalg.eig(cov)

print("Transformed Space")

print(np.matmul(X, V[:,np.argmax(D)]))

print("Variance Preserved by Highest Eigenvalue: " + str(np.max(D)/np.sum(D)))

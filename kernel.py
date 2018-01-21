import math
import numpy as np

def linear(X1, X2):
    return np.dot(X1,X2.T)

def polinomial(X1, X2, degree=2):
    return np.power(np.dot(X1, X2.T), degree)

def rbf(X1, X2, sigma=1):
    normalization_coef= 1.0/ (sigma * math.sqrt(2*math.pi))
    norm2=np.zeros((len(X1),len(X2)))
    for i in range(len(X1)):
        for j in range(len(X2)):
            norm2[i,j] = np.linalg.norm(X1[i]-X2[j],2)
    return normalization_coef *np.exp(-np.power(norm2,2)*sigma/2)




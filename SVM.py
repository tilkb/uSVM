import numpy as np
from kernel import linear
from SMO import SMO

class SVM:
    def __init__(self, C=1, kernel=linear):
        self.C = C
        self.kernel = kernel
        self.support_vectors = np.array([[]])
        self.support_label = np.array([[]])
        self.alfa=np.array([[]])

    def fit(self, X, Y):
        solver = SMO(self.kernel(X,X),Y,self.C)
        self.alfa = solver.solve()
        print(self.alfa)
        
    def predict(self, X):
        gram = self.kernel(self.support_vectors, X).T
        weights = np.multipy(self.alfa,self.support_label)
        np.sign(np.dot(gram,weight))
        
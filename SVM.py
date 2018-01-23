import numpy as np
from kernel import linear
from SMO import SMO

class SVM:
    def __init__(self, C=1, kernel=linear):
        self.C = C
        self.bias=0
        self.kernel = kernel
        self.support_vectors = np.array([[]])
        self.support_label = np.array([[]])
        self.alfa=np.array([[]])

    def fit(self, X, Y):
        solver = SMO(self.kernel(X,X),Y,self.C)
        self.alfa = solver.solve()
        self.support_vectors=X
        self.support_label=Y
        self.bias=0
        count=0
        for i in range(len(self.alfa)):
            if self.alfa[i]>0.01:
                count=count+1
                self.bias=self.bias+ self.support_label[i] * self.alfa[i]*self.kernel(self.support_vectors[i],self.support_vectors[i])
        self.bias = self.bias / count
    def predict(self, X):
        gram = self.kernel(self.support_vectors, X)
        weights = np.multiply(self.alfa,self.support_label)
        return  np.sign(np.dot(gram,weights)-self.bias)
        
import numpy as np
from kernel import linear
from SVM import SVM

X=np.array([[1,1],[0,0.5],[3,3],[5,5]])
Y=np.array([-1,-1,1,1])

svm=SVM()

svm.fit(X,Y)
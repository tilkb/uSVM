import numpy as np
from kernel import linear
from SVM import SVM

X=np.array([[0,1],[1,2],[1,3],[0,4]])
Y=np.array([-1,-1,-1,1])

svm=SVM(C=1)

svm.fit(X,Y)
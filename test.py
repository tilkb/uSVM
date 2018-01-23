import numpy as np
from kernel import linear
from SVM import SVM
from sklearn.svm import SVC

X=np.array([[0,1],[2,2],[2,1],[1,0]])
X2= np.array([[0,1],[2,2],[2,3],[0,4]])
Y=np.array([-1,-1,-1,1])

svm=SVM(C=1)
svm.fit(X,Y)

print(svm.predict(X))
print(svm.alfa)

original_SVM=SVC()
original_SVM.fit(X,Y)
print(original_SVM.dual_coef_)
import numpy as np

class SMO:
    def __init__(self, gram_matrix, labels, C ):
        self.C=C
        self.labels=labels
        self.K=gram_matrix
        #no supportvector is a correct initialization
        self.alfa= np.zeros(len(labels))
        self.bias =0

    def calc_obj(self):
        label_array=self.labels.reshape(1,-1)
        alfa_array=self.alfa.reshape(1,-1)
        gram_label=np.dot(label_array.T,label_array)
        gram_alfa=np.dot(alfa_array.T,alfa_array)
        return np.sum(self.alfa)-0.5*np.sum(np.multiply(self.K,np.multiply(gram_label,gram_alfa)))

    def search_max(self,nu,index1,index2):

        #remove index1 and index2
        iter1=np.multiply(np.multiply(self.alfa,self.labels),self.K[index1,:])
        iter1 = self.labels[index2]*(np.sum(iter1)-iter1[index1]-iter1[index2])
        iter2=np.multiply(np.multiply(self.alfa,self.labels),self.K[index2,:])
        iter2 = self.labels[index2]*(np.sum(iter2)-iter2[index1]-iter2[index2])

        result= 1-self.labels[index1]*self.labels[index2] + self.K[index1,index1] * nu * self.labels[index2]-self.K[index1,index2]*self.labels[index2] * nu + 0.5 *iter1 - 0.5 * iter2 
        
        normalized_result =result / (self.K[index1,index1]+self.K[index2,index2] - 2*self.K[index1,index2])
        return normalized_result 
                                                 

    def step(self, index1, index2):
        """
        index1 and index2 are the selected points
        """
        if index1 == index2:
            return
        nu = self.alfa[index1]*self.labels[index1]+self.alfa[index2]*self.labels[index2]

        #determine limits
        if self.labels[index1] != self.labels[index2]:
            high=min(self.C,self.C+self.alfa[index2]-self.alfa[index1])
            low = max(self.C,self.alfa[index2]-self.alfa[index1])
        else:
            high=min(self.C,self.alfa[index1]+self.alfa[index2])
            low = max(0, self.alfa[index1]-self.alfa[index2]-self.C)


        alfa2=self.search_max(nu,index1,index2)
        if alfa2>high:
            alfa2=high
        elif alfa2<low:
            alfa2=low


        #change variables
        before = self.calc_obj()
        tmp1=self.alfa[index2]
        tmp2=self.alfa[index1]
        self.alfa[index2] = alfa2
        self.alfa[index1] = (nu-alfa2*self.labels[index2])*self.labels[index1]
        if before>self.calc_obj():
            self.alfa[index2]=tmp1
            self.alfa[index1]=tmp2
            self.search_max(nu,index1,index2)
            
        
    def solve(self, maxiter=100):
        nr_iter=0
        #differance between iteraions
        while nr_iter<maxiter:
            for i in range(len(self.alfa)*2):
                self.step(np.random.randint(0,len(self.alfa)),np.random.randint(0,len(self.alfa)))
            nr_iter=nr_iter + 1

        return self.alfa
        

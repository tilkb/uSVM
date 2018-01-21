import numpy as np

class SMO:
    def __init__(self, gram_matrix, labels, C ):
        self.C=C
        self.labels=labels
        self.K=gram_matrix
        #no supportvector is a correct initialization
        self.alfa= np.zeros(len(labels))        

    def calc_obj(self):
        label_array=self.labels.reshape(1,-1)
        alfa_array=self.alfa.reshape(1,-1)
        gram_label=np.dot(label_array.T,label_array)
        gram_alfa=np.dot(alfa_array.T,alfa_array)
        np.multiply(gram_label,gram_alfa)
        return np.sum(self.alfa)-0.5*np.sum(np.multiply(self.K,np.multiply(gram_label,gram_alfa)))

    def search_max(self,nu,index1,index2):
        #remove index1 and index2
        iter1=np.multiply(np.multiply(self.alfa,self.labels),self.K[index2,:])
        iter1 = self.labels[index2]*(np.sum(iter1)-iter1[index1]-iter1[index2])

        iter2=np.multiply(np.multiply(self.alfa,self.labels),self.K[index1,:])
        iter2 = self.labels[index2]*(np.sum(iter1)-iter2[index1]-iter2[index2])
    

        result=-1.0/(2.0*self.K[index1,index2])*(1+nu*self.labels[index1]
                -self.labels[index1]*self.labels[index2]-self.K[index1,index2]*self.labels[index2]*nu
                -0.5*iter1-0.5*iter2)
        return result
                                                 

    def step(self, index1, index2):
        """
        index1 and index2 are the selected points
        """
        if index1 == index2:
            return
        nu = self.alfa[index1]*self.labels[index1]+self.alfa[index2]*self.labels[index2]

        #determine limits
        high =(nu-self.C*self.labels[index1])/self.labels[index2]
        low =nu/self.labels[index2]
        #derivative can be negative:
        if high<low:
            tmp=high
            high=low
            low=tmp
        if high>self.C:
            high = self.C
        if low<0:
            low = 0

        alfa2=self.search_max(nu,index1,index2)
        print(alfa2)
        print((nu-alfa2*self.labels[index2])*self.labels[index1])
        if alfa2>high:
            alfa2=high
        elif alfa2<low:
            alfa2=low

        #change variables
        self.alfa[index2] = alfa2
        self.alfa[index1] = (nu-alfa2*self.labels[index2])*self.labels[index1]
            
        
    def solve(self, tolerance=0.01, maxiter=1000):
        nr_iter=0
        #differance between iteraions
        diff=tolerance+1.0
        before=self.calc_obj()
        while nr_iter<maxiter and tolerance<diff:
            #select two item
            index = np.random.permutation(len(self.alfa))
            for i in range(len(self.alfa)//2):
                self.step(index[i],index[-1-i]) 

            after=self.calc_obj()
            diff=before-after
            before=after
            nr_iter=nr_iter + 1
            print(after)

        return self.alfa
        

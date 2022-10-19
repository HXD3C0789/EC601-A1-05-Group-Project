import numpy as np
class PCA:
    def __init__(self,x):
        self.x=x
    def SVDdecompose(self):
        u,s,v=np.linalg.svd(self.x,full_matrices=False)
        self.lamda=s**2 # dot(S)
        self.p=v.T
        self.t=u*s
        compare=self.lamda[:-1]/self.lamda[1:]
        return compare

    def PCAcompose(self,k):
        p=self.p[:,:k]
        t=self.t[:,:k]   #t是x在另一个空间的线性变换，可以代表x且顺序没有变
        return t,p
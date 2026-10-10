import numpy as np
class Vector:
    def __init__(self, v:np.array):
        self.vector=v
    def add(self, v:"Vector"):
        """
        Adds the vector v to self
        """
        self.vector=np.add(self.vector,v.vector)

    def substract(self, v:"Vector"):
        """
        substracts the vector v to self 
        """
        self.vector=np.subtract(self.vector,v.vector)

    def scale(self,alpha):
        """
        scale "self" to alpha
        """

        self.vector=self.vector*alpha

    def norm(self):
        """
        returns the norm of the vector
        """

        return np.linalg.norm(self.vector)
    
    
          

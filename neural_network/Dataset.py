class Dataset:
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, key):
        return (self.X[key],self.Y[key])
        
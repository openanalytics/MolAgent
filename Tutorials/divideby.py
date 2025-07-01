import numpy as np

class DivideBy():
    """!
    Divide by transformation for the predictions
    """
    def __init__(self,value:float=2.0):
        assert np.abs(value)>1e-16, 'do not divide by 0'
        self.value=value
    
    def __call__(self,array:np.ndarray=None):
        assert len(array)>0, 'empty array'
        return array/self.value
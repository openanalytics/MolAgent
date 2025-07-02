import numpy as np
class HammingMultioutputScore():   
    def __call__(self,y_true,y_pred):
        return -1*np.sum(np.not_equal(y_true, y_pred))/float(y_true.size)

    def __name__(self):
        return 'HammingMultioutputScore'
"""implementation of the featurewise dimensionality reduction.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""

from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
import re
from sklearn.base import clone


class FeatureTypeDimReduction(TransformerMixin, BaseEstimator):
    """
    A scikit transformer that applies dimensionality reduction method on different column ranges in X. You can specify and optimize which 
    dimensionality reduction method on which set of features based on the given list of column splits (e.g. column indices).
    
    The parameters can be set using fs<i>__fs_method to set the method for subset i and fs<i>__parameter to set a parameter of method i. 
    """
    def __init__(
        self,
        estimators,
        col_splits,
        names
    ):
        """
        Initialization
        
        Args:
            estimators: list of estimators
            col_splits: list of column indices to split the data matrix
            names: list of names of feature types
        """
        assert(len(estimators)==len(col_splits)-1), 'length of estimators is not equal to length of col splits-1'
        assert(len(estimators)==len(names)), 'length of estimators is not equal to length of given names'
        ## list of estimators
        self.estimators = estimators
        ## list of column indices to determine feature splits
        self.col_splits = col_splits
        ## list of names of feature types
        self.names=names
        
    def transform(self, X):
        """
        Transform the given matrix X using the estimators on the data splits defined in col_splits
        
        Args:
            X: feature matrix X
        
        Returns:
            transformed feature matrix X with reduced dimension (potentially)
        """
        col_list = []
        for index in range(len(self.col_splits)-1):
            if self.estimators[index] is not None and self.estimators[index]!='passthrough':
                col_list.append(self.estimators[index].transform(X[:, self.col_splits[index]:self.col_splits[index+1]]))
            else:
                col_list.append(X[:, self.col_splits[index]:self.col_splits[index+1]])
        new_X=np.concatenate(col_list, axis=1)
        if new_X.shape[1]==0:
            new_X=np.ones((new_X.shape[0], 1))
        return new_X

    def fit(self, X, y=None,**fit_params):
        """
        fits the different estimators using the given matrix
        
        Args:
            X: data matrix
            y; target vector
            fit_params: dictionary of fit params (not used)
        """
        for index in range(len(self.col_splits)-1):
            if self.estimators[index] is not None and self.estimators[index]!='passthrough':
                self.estimators[index].fit(X[:, self.col_splits[index]:self.col_splits[index+1]],y)
        return self
    
    def set_params(self, **parameters):
        """
        sets parameters for the individual estimators using the prefix fs<i>__ with i the index of the estimator
        
        The max number of selected features is trimmed to the max available features if the given exceeds this value. This is done by searching the parameters for n_components, k and max_features. This is based on the available methods in sklearn. 
        
        Args:
            parameters: dictionary of parameters
        """
        param_lists=[{} for i in range(len(self.estimators))]
        r = re.compile(f"(.*)(n_components|k|max_features)")
        key_list=list(filter(r.match, parameters.keys()))
        for key in key_list:
            index = int(re.findall('[0-9]+', key)[0])
            if parameters[key]>self.col_splits[index+1]-self.col_splits[index]:
                parameters[key]=self.col_splits[index+1]-self.col_splits[index]
        for parameter, value in parameters.items():
            for i in range(len(self.estimators)):
                if parameter.startswith(f'fs{i}'):
                    if parameter[len(f'fs{i}__'):]=="":
                        if value =='passthrough':
                            self.estimators[i]=None
                        else:
                            #clone seems required since the same estimator is possibly provided for multiple featuretypes
                            self.estimators[i]=clone(value)
                    else:
                        param_lists[i][parameter[len(f'fs{i}__'):]]=value
        for i in range(len(self.estimators)):
            if self.estimators[i] is not None and self.estimators[i]!='passthrough':
                self.estimators[i].set_params(**param_lists[i])
        return self
    
    def __str__(self):
        return "Dimensionality Reduction ("+', '.join([str(nm)+"->"+str(m) for nm,m in zip(self.names,self.estimators)]  )+")"
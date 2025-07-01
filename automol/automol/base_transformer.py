"""BaseEstimatorTransformer transforms data into concatenated predictions of the base estimators.

This class is designed to transform a data matrix using a list of estimators. The output of these estimators is concatenated in a new and thus transformed feature matrix X_. In the case of classification, the predicted probabilities for each class are taken. For regression, it is simply the output of the estimator. 

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""

from typing import List

######################################################
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
#scikit transformer that takes the features matrix as input and outputs the predictions of the list of models.
#Used as a pipeline step for a plain top level blender (do not use for stacking model!)
class BaseEstimatorTransformer(BaseEstimator,TransformerMixin):
    """
    Scikit transformer that takes the feature-matrix as input and outputs the predictions of the list of models.
    """
    def __init__(self, model_list:List,classification:bool=False,nb_props:int=0):
        """ Initialization
        
        Args:
            model_list [List]: List of estimators. 
            classification [bool]: Boolean to indicate classifiers.
        """
        
        ## list of provided estimators
        self.model_list=model_list
        ## number of models
        self.n_models=len(model_list)
        ## boolean to indicate classification models
        self.classification=classification
        self.nb_targets=-1
        assert nb_props>=0, 'nb_props negative'
        self.nb_props=nb_props
        
    def fit(self,X,y=None,**fit_params):
        """
        fits the list of estimators given X and y
        
        Args:
            X: the data matrix X
            y: the target y
            fit_params: dictionary of fit params
        """
        if self.nb_props>0:
            assert self.nb_props<X.shape[1]
            Xt=X[:,:-self.nb_props]
        else:
            Xt=X
        if len(y.shape)==2:
            self.nb_targets=y.shape[1]
        else:
            self.nb_targets=1
        [m.fit(Xt,y,**fit_params) for m in self.model_list]
        if self.classification: 
            ## number of classes
            if self.nb_targets>1:
                self.nb_proba_vectors=0
                for i in range(self.nb_targets):
                    self.nb_proba_vectors+=int(np.nanmax(y[:,i])+1)
            else:
                 self.nb_proba_vectors=int(np.nanmax(y)+1)                
        return self
        
    def transform(self,X,y=None):
        """
        Returns the outputs the predictions of the base estimators for given data X
        
        For regression, the values
        For classification, the predicted probabilities for all classes
        
        Args:
            X: the data matrix
            y: the target
            
        Returns:
            transformed matrix X_
        """
        if self.nb_props>0:
            assert self.nb_props<X.shape[1]
            Xt=X[:,:-self.nb_props]
            X_p=X[:,-self.nb_props:]
        else:
            Xt=X
            X_p=None

        if self.classification:
            X_=np.zeros((Xt.shape[0],self.n_models*self.nb_proba_vectors))
            x_index=0
            for i,m in enumerate(self.model_list):
                proba_l=m.predict_proba(Xt)
                if isinstance(proba_l, list):
                    proba_l=np.concatenate(proba_l, axis=1)
                for j in range(self.nb_proba_vectors):
                    X_[:,x_index]= proba_l[:,j]
                    x_index+=1
        else:
            X_=np.zeros((Xt.shape[0],self.n_models*self.nb_targets))
            for i,m in enumerate(self.model_list):
                m_pred=m.predict(Xt)
                if len(m_pred.shape)==1:
                    X_[:,self.nb_targets*i:self.nb_targets*(i+1)]=np.expand_dims(m_pred, axis=1)
                elif self.nb_targets==1:
                    X_[:,self.nb_targets*i:self.nb_targets*(i+1)]=m_pred
                else:
                    X_[:,self.nb_targets*i:self.nb_targets*(i+1)]=m_pred.squeeze()
        if X_p is not None: X_=np.hstack((X_,X_p))
        return X_
    def __str__(self):
        CEND      = '\33[0m'
        CBOLD     = '\33[1m'
        return CBOLD+"Base Estimators:"+CEND+"\n"+'; '.join([str(m.steps[1][1])+':'+str(m.steps[2][1]) if m.steps[1][1]!='passthrough' else str(m.steps[2][1]) for m in self.model_list]  ) +"\n"+ CBOLD+"Blender"+CEND
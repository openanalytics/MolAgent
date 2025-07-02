""" implementation of the multi-layer perceptron wrappers.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.base import BaseEstimator, ClassifierMixin,RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


"""
Wrappers for the sklearn MLPClassifier,MLPRegressor methods
"""
class MLPClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for MLPClassifier
    """
    def __init__(self, hidden_layers=2,hidden_layers_size=30, activation='relu',solver='adam',alpha=1e-4,batch_size='auto',learning_rate='constant',
                 learning_rate_init=1e-3, max_iter=200, random_state=42,validation_fraction=0.15):
        """
        Initialization, see the sklearn documentation on the MLPClassifier
        
        Args:
            hidden_layers: number of hidden layers
            hidden_layers_size: size of the hidden layers
            activation: activation function
            solver: optimization
            alpha: alpha
            batch_size: batch size
            learning_rate: learning rate
            learning_rate_init: initial learning rate
            max_iter: max number of iterations
            random_state: random state
            validation_fraction: fraction of validation
        """
        ## number of hidden layers
        self.hidden_layers = hidden_layers
        ## size of the hidden layers
        self.hidden_layers_size = hidden_layers_size
        ## activation function
        self.activation = activation
        ## network parameter optimizer
        self.solver = solver
        ## alpha parameter
        self.alpha=alpha
        ## batch size
        self.batch_size=batch_size
        ## learning rate
        self.learning_rate=learning_rate
        ## initial learning rate
        self.learning_rate_init=learning_rate_init
        ## max number of iterations
        self.max_iter=max_iter
        ## random state
        self.random_state=random_state
        ## validation fraction
        self.validation_fraction=validation_fraction
        
    def fit(self, X, y,**fit_params):
        """
        fits the mlpclassifier
        
        Args:
            X: the data matrix
            y: the target vector
            fit_params: dictionary with fitting parameters
        
        Returns:
            self
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.clf = MLPClassifier(hidden_layer_sizes=tuple(self.hidden_layers_size for i in range(self.hidden_layers) ), activation=self.activation,solver=self.solver,
                            alpha=self.alpha,batch_size=self.batch_size, learning_rate=self.learning_rate,
                            learning_rate_init=self.learning_rate_init, max_iter=self.max_iter , random_state=self.random_state,
                            early_stopping=True,validation_fraction=self.validation_fraction).fit(X, y,**fit_params)
        
        return self
    
    def predict(self, X):
        """
        call the predict function of the mlpclassifier
        
        Args:
            X: data matrix X
        
        Returns:
            the predictions
        """
        check_is_fitted(self.clf)
        X = check_array(X)
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        """
        calls the predict_proba function of the mlpclassifier
        
        Args:
            X: data matrix X
        
        Returns:
            the probabilistic predictions
        """
        check_is_fitted(self.clf)
        X = check_array(X)
        return self.clf.predict_proba(X)
    
    def set_params(self, **parameters):
        """
        sets the parameters
        
        Args:
            parameters: the parameter dictionary
        
        Returns:
            self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class MLPRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper for MLPRegressor
    """
    
    def __init__(self, hidden_layers=2,hidden_layers_size=30, activation='relu',solver='adam',alpha=1e-4,batch_size='auto',learning_rate='constant',
                 learning_rate_init=1e-3, max_iter=200, random_state=42,validation_fraction=0.15):
        """
        Initialization, see the sklearn documentation on the MLPRegressor
        
        Args;
            hidden_layers: number of hidden layers
            hidden_layers_size: size of the hidden layers
            activation: activation function
            solver: optimization
            alpha: alpha
            batch_size: batch size
            learning_rate: learning rate
            learning_rate_init: initial learning rate
            max_iter: max number of iterations
            random_state: random state
            validation_fraction: fraction of validation
        """
        ## number of hidden layers
        self.hidden_layers = hidden_layers
        ## size of the hidden layers
        self.hidden_layers_size = hidden_layers_size
        ## activation function
        self.activation = activation
        ## network parameter optimizer
        self.solver = solver
        ## alpha parameter
        self.alpha=alpha
        ## batch size
        self.batch_size=batch_size
        ## learning rate
        self.learning_rate=learning_rate
        ## initial learning rate
        self.learning_rate_init=learning_rate_init
        ## max number of iterations
        self.max_iter=max_iter
        ## random state
        self.random_state=random_state
        ## validation fraction
        self.validation_fraction=validation_fraction
                
    def fit(self, X, y,**fit_params):
        """
        fits the mlpregressor
        
        Args:
            X: the data matrix
            y: the target vector
            fit_params: dictionary with fitting parameters
        
        Returns:
            self
        """
        X, y = check_X_y(X, y)
        self.reg = MLPRegressor(hidden_layer_sizes=tuple(self.hidden_layers_size for i in range(self.hidden_layers) ), activation=self.activation,solver=self.solver,
                            alpha=self.alpha,batch_size=self.batch_size, learning_rate=self.learning_rate,
                            learning_rate_init=self.learning_rate_init, max_iter=self.max_iter , random_state=self.random_state,
                            early_stopping=True,validation_fraction=self.validation_fraction).fit(X, y,**fit_params)
        return self
    
    def predict(self, X):
        """
        call the predict function of the MLPRegressor
        
        Args:
            X: data matrix X
        
        Returns:
            the predictions
        """
        check_is_fitted(self.reg)
        X = check_array(X)
        return self.reg.predict(X)
    
    def set_params(self, **parameters):
        """
        sets the parameters
        
        Args:
            parameters: the parameter dictionary
        
        Returns:
            self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
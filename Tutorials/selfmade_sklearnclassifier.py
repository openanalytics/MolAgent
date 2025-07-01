"""
A wrapper around the sklearn MLPClassifier. 

Basicly any sklearn classifier used in AutoML requires the following functions:

fit(self, X, y,**fit_params)
predict(self, X)
predict_proba(self, X)
set_params(self, **parameters)

additionaly the __init__ function must directly set its attributes no alternation of the provided values. 

Note that this is just an example and this method is provided by JNJ AutoML internally, for deployment use that one ;)
"""

from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class MLPClassifierWrapper(BaseEstimator, ClassifierMixin):
    
    def __init__(self, hidden_layers=2,hidden_layers_size=30, activation='relu',solver='adam',alpha=1e-4,batch_size='auto',learning_rate='constant',
                 learning_rate_init=1e-3, max_iter=200, random_state=42):
        self.hidden_layers = hidden_layers
        self.hidden_layers_size = hidden_layers_size
        self.activation = activation
        self.solver = solver
        self.alpha=alpha
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.learning_rate_init=learning_rate_init
        self.max_iter=max_iter
        self.random_state=random_state
        
    def fit(self, X, y,**fit_params):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.clf = MLPClassifier(hidden_layer_sizes=tuple(self.hidden_layers_size for i in range(self.hidden_layers) ), activation=self.activation,solver=self.solver,
                            alpha=self.alpha,batch_size=self.batch_size, learning_rate=self.learning_rate,
                            learning_rate_init=self.learning_rate_init, max_iter=self.max_iter , random_state=self.random_state,
                            early_stopping=True,validation_fraction=0.15).fit(X, y,**fit_params)
        
        return self
    
    def predict(self, X):
        check_is_fitted(self.clf)
        X = check_array(X)
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        check_is_fitted(self.clf)
        X = check_array(X)
        return self.clf.predict_proba(X)
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
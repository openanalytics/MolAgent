"""implementation of different optimization methods for parameter search.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""
 

import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone

import warnings
import logging

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


from psutil import cpu_count

######################################################
class ParamSearch: 
    """
    A common interface between the different methods to find the best parameters. 
    
    Differentiates between RandomizedSearch, GridSearch and HyperoptSearch.
    """
    
    def __init__(self):
        """
        Initialization
        """
        ## the best estimator found by parameter optimization
        self.best_estimator=None
        ## dictionary of fitting parameters
        self.fit_params=None
        ## the found parameters
        self.best_params_=None
        ## the index of the parameters
        self.best_index_=None
        ## dictionary of the results
        self.cv_results_={}
        
    def clear_fit_params(self):
        """
        clears all the fitting parameters
        """
        self.best_estimator=None
        self.fit_params=None
        self.best_params_=None
        self.best_index_=None
        self.cv_results_={}
        
    def add_attribute_fit_params(self,key,value):
        """
        add an attribute to the fit_params
        
        Args:
             key: key in the fit_params dictionary
             value: value corresponding to the key
        """
        if self.fit_params is None:
            self.fit_params={key:value}
        else:
            self.fit_params[key]=value

    def search(self,estimator=None,params=None, scoring=None,cv=None,X_train=None, y_train=None, random_state=42):
        """
        finds the best parameters using the specified scoring options and given params
        
        Args:
             estimator: the estimator
             params: the dictionary containing the method parameter
             scoring: the scorer for scikit
             cv: cross-validation scikit object representing the folds
             X_train: data matrix to be fitted
             y_train: observations to be fitted
             random_state: the random_state integer value
        """
        pass
    
    def get_best_estimator(self):
        """
        returns the best estimator if the search method has been called
        
        Returns:
            the found estimator
        
        """
        return self.best_estimator
        
class GridSearch(ParamSearch):
    """
    Implementation of the common interface between the different methods to find the best parameters for GridSearch. 
    """
    
    def __init__(self,refit=True,n_jobs=-1, verbose=False):
        """
        Initialization
        
        Args:
             refit: boolean to set refit option in GridSearchCV
             n_jobs: number of jobs used in GridSearchCV
             verbose: indicate use of print statements
        """
        super(GridSearch,self).__init__()
        ## boolean to refit
        self.refit=refit
        ## number of threads/jobs
        self.n_jobs=n_jobs
        ## indicate use of print statements
        self.verbose=verbose
        
    def search(self,estimator=None,params=None, scoring=None,cv=None,X_train=None, y_train=None, random_state=42):
        """
        finds the best parameters using the specified scoring options and given params
        
        Args:
             estimator: the estimator
             params: the dictionary containing the method parameter
             scoring: the scorer for scikit
             cv: cross-validation scikit object representing the folds
             X_train: data matrix to be fitted
             y_train: observations to be fitted
             random_state: the random_state integer value
        """
        
        grid = GridSearchCV(estimator, param_grid=params, scoring=scoring,cv=cv,
                                        verbose=self.verbose, n_jobs=self.n_jobs #,iid =False
                                        ,refit=self.refit)
        if self.fit_params is not None and isinstance(estimator, Pipeline):
            grid.fit(X_train, y_train,**self.fit_params)
        else:
            grid.fit(X_train, y_train)
        if self.verbose: print(f'Best score after GridSearchCV: {grid.best_score_}')
        self.best_params_=grid.best_params_
        self.best_index_=grid.best_index_
        self.cv_results_=grid.cv_results_
        self.best_estimator=grid.best_estimator_

class RandomizedSearch(ParamSearch):
    """
    Implementation of the common interface between the different methods to find the best parameters for RandomizedSearch. 
    """
    
    def __init__(self,refit=True,n_jobs=-1, verbose=False,n_iter=60):
        """
        Initialization
        
        Args:
             refit: boolean to set refit option in RandomizedSearchCV
             n_jobs: number of jobs used in RandomizedSearchCV
             verbose: indicate use of print statements
             n_iter: number of iterations used in RandomizedSearchCV
        """
        super(RandomizedSearch,self).__init__()
        ## refit option in RandomizedSearchCV
        self.refit=refit
        ## number of threads used in RandomizedSearchCV
        self.n_jobs=n_jobs
        ## print statements?
        self.verbose=verbose
        ## number of iterations in RandomizedSearchCV
        self.n_iter=n_iter
        
    def search(self,estimator=None,params=None, scoring=None,cv=None,X_train=None, y_train=None, random_state=42):
        """
        finds the best parameters using the specified scoring options and given params
        
        Args:
             estimator: the estimator
             params: the dictionary containing the method parameter
             scoring: the scorer for scikit
             cv: cross-validation scikit object representing the folds
             X_train: data matrix to be fitted
             y_train: observations to be fitted
             random_state: the random_state integer value
        """
        grid = RandomizedSearchCV(estimator, param_distributions=params,n_iter=self.n_iter, scoring=scoring,cv=cv,
                                        verbose=self.verbose, n_jobs=self.n_jobs #,iid =False
                                        ,refit=self.refit,random_state=random_state)
        if self.fit_params is not None and isinstance(estimator, Pipeline):
            grid.fit(X_train, y_train,**self.fit_params)
        else:
            grid.fit(X_train, y_train)
        if self.verbose: print(f'Best score after RandomizedSearchCV: {grid.best_score_}')
        self.best_params_=grid.best_params_
        self.best_index_=grid.best_index_
        self.cv_results_=grid.cv_results_
        self.best_estimator=grid.best_estimator_
        
class HyperoptSearch(ParamSearch):
    """
    Implementation of the common interface between the different methods to find the best parameters for HyperoptSearch. 
    """
    
    def __init__(self,algo=tpe.suggest,n_iter=60,n_jobs=-1):
        """
        Initialization
        
        Args:
             algo: algorithm used in Hyperopt fmin optimization
             n_jobs: number of jobs used in Hyperopt fmin optimization
             n_iter: number of iterations used in Hyperopt fmin optimization
        """
        super(HyperoptSearch,self).__init__()
        ## hyperopt algorithms
        self.algo=algo
        ## number of iterations
        self.n_iter=n_iter
        if n_jobs<0: n_jobs=cpu_count(logical=False)
        ## number of threads
        self.n_jobs=n_jobs
        ## boolean to check for parallization
        self.use_spark=n_jobs>1
        ## number of jobs if pyspark is not available
        self.cross_val_jobs=1
        ## boolean to show progressbar
        self.show_progressbar=not n_jobs>1
        
    def search(self,estimator=None,params=None, scoring=None,cv=None,X_train=None, y_train=None, random_state=42):
        """
        finds the best parameters using the specified scoring options and given params
        
        Args:
             estimator: the estimator
             params: the dictionary containing the method parameter
             scoring: the scorer for scikit
             cv: cross-validation scikit object representing the folds
             X_train: data matrix to be fitted
             y_train: observations to be fitted
             random_state: the random_state integer value
        """
        if isinstance(params,list):
            #optimizing single method
            if len(params)==1:
                search_space=params[0]
            #optimizing multiple methods
            else:
                search_space=hp.choice('methods', params)

        if self.use_spark:
            try:
                from pyspark import SparkContext
                from hyperopt import SparkTrials
                trials_obj = SparkTrials(parallelism=self.n_jobs)
                self.cross_val_jobs=1
            except ImportError as e:
                warnings.warn("Error when importing pyspark, using sequential hyperopt optimization", ImportWarning)
                print('Warning: error when importing pyspark, using sequential hyperopt optimization')
                self.cross_val_jobs=self.n_jobs
                trials_obj = hyperopt.Trials()
                
            #logging.basicConfig(level=logging.ERROR)
            #from pyspark import SparkContext
            #sc = SparkContext.getOrCreate()
            #sc.setLogLevel("ERROR")
            #logger = sc._jvm.org.apache.log4j
            #logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)
        else:
            trials_obj = hyperopt.Trials()        
        
        def objective(par):
            changed_estimator=clone(estimator)
            changed_estimator.set_params(**par)
            if self.fit_params is not None and isinstance(changed_estimator, Pipeline):
                cv_score = cross_val_score(changed_estimator, X_train, y_train,scoring=scoring,cv=cv,fit_params=self.fit_params,n_jobs=self.cross_val_jobs)
            else:
                cv_score = cross_val_score(changed_estimator, X_train, y_train,scoring=scoring,cv=cv,n_jobs=self.cross_val_jobs)
            return {'loss': -cv_score.mean(), 'status': STATUS_OK, 'cv_score': cv_score}

        loggers_to_shut_up = [
            "hyperopt.tpe",
            "hyperopt.fmin",
            "hyperopt.pyll.base",
            "py4j.clientserver",
            "hyperopt-spark",
        ]

        
        for logger in loggers_to_shut_up:
            logging.getLogger(logger).setLevel(logging.ERROR)
            
        rstate=np.random.default_rng(random_state) 
        
        try: 
            best_result = fmin(
                fn=objective, 
                space=search_space,
                algo=self.algo,
                max_evals=self.n_iter, 
                show_progressbar=self.show_progressbar,
                trials=trials_obj,
                rstate=rstate)
        except RuntimeError as e:
            print(f'RuntimeError: {e}. Trying with sequential optimization.')
            #force sequential 
            self.cross_val_jobs=self.n_jobs
            trials_obj = hyperopt.Trials()
            best_result = fmin(
                fn=objective, 
                space=search_space,
                algo=self.algo,
                max_evals=self.n_iter, 
                show_progressbar=self.show_progressbar,
                trials=trials_obj,
                rstate=rstate)
        
        #only storing best values and setting best index to zero
        best_trial = trials_obj.best_trial
        best_cv_scores=best_trial['result']['cv_score']
        self.cv_results_={}
        for fold,score in enumerate(best_cv_scores):
            self.cv_results_[f'split{fold}_test_score']=[score]
        self.best_params_=hyperopt.space_eval(search_space, best_result)
        self.best_index_=0
        estimator.set_params(**self.best_params_)
        self.best_estimator=estimator
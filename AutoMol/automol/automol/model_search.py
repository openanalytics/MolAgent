"""implementation of functionality to perform nested cross-validation for all the different method configurations.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""
import numpy as np, math

from .stacking_methodarchive import *
from .base_transformer import BaseEstimatorTransformer
from sklearn.model_selection import StratifiedKFold , GroupKFold ,StratifiedKFold ,LeaveOneGroupOut
from sklearn.ensemble import StackingRegressor,StackingClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from tempfile import mkdtemp
from shutil import rmtree
from joblib import Memory
import time
import re

######################################################
class NestedCVModelSearch:
    """
    NestedCVModelSearch searches for the best parameters for the estimators using nested cross-validations and works closely together with ModelFinder (determines classification/regression functionality) which in turn is called by FeatureGenerationRegressor.
    
    NestedCVModelSearch finds methods using nested cross-validation functions which are called from ModelFinder.
    The inner folds finds for each method type the best parameters for each fold (parameters for one method may differ over the folds)
    The outer folds can be used to find a top level method (blender, stacking) using the models found in the inner folds
    """
    def __init__(self,verbose,top_stacking=False):
        """
        Initialization
        
        Args:
            verbose: boolean to set for print statements
            top_stacking: boolean to set to use a scikit stacking model for the top estimator
        """
        ## boolean to indicate verbosity
        self.verbose=verbose
        ## boolean to set for top stacking method configuration
        self.top_stacking=top_stacking
    
    def get_folds(self,X ,y,groups,cv=5,split='SKF'):
        """
        split data in folds using given data X and cross-validation method
        
        Args:
            X: data matrix X
            y: the y-values, te.g. target/observations
            groups: the non-overlapping groups of the data
            cv: integer to set value of k in kfold or stratified kfold
            split: string to indicate which splitting method to be used.
               LGO = LeaveOneGroupOut CV
               GKF = Using GroupKFold CV
               SKF = Stratified k-fold CV
        """
        if split== 'LGO':
            group_kfold = LeaveOneGroupOut().split(X, y, groups)
            n_folds=LeaveOneGroupOut().get_n_splits(groups=groups)
            if self.verbose: print('Using LeaveOneGroupOut CV' )
        elif split== 'GKF':
            group_kfold = GroupKFold(n_splits=cv).split(X, y, groups)
            n_folds=GroupKFold(n_splits=cv).get_n_splits(groups=groups)
            if self.verbose: print('Using GroupKFold CV' )
        else: #'SKF'
            group_kfold =StratifiedKFold(n_splits=cv).split(X, groups)
            n_folds=StratifiedKFold(n_splits=cv).get_n_splits(groups=groups)
            if self.verbose:print('Stratified k-fold CV' )
        
        return group_kfold,n_folds
    
    #implements inner fold parameter search using given ModelFinder, training and test set 
    def inner_fold_search(self,modfinder,X_train ,y_train,X_test ,y_test, out, paramsearch, scoring, inner_group_kfold, cv, params_grid, ests_list, use_memory, prefix_dict=None, random_state=42):
        """
        find models using inner folds of nested cross-validation
        
        Args:
            modfinder: ModelFinder object to differentiate between classification and regression functionality such as scoring output
            X_train: training data
            y_train: training observations
            X_test: test data
            y_test: test observations
            out: output dictionary
            paramsearch: a derived ParamSearch object that performs the parametersearch
            scoring: scorer string optimized in the parameter search
            inner_group_kfold: inner folds
            cv: number of folds 
            params_grid: list of parameters options
            ests_list: list of already fitted/found base estimators
            use_memory: boolean to indidcate use of memory 
            prefix_dict: dictionary of prefixes 
            random_state: random state parameter 
        
        Returns:
            dictionary with the found models inside key 'models'    
        """
        #iterate over methods in params_grid (the method is specified in the options)
        for o_par in params_grid:
            if isinstance(o_par, (dict)):    o_par= [o_par]
            par= [e.copy() for e in o_par]
            assert isinstance(par, (list, tuple))
            if self.verbose:
                print('****************')
                print('Searching models using the params grid of ',par[0][prefix_dict['method_prefix']][0].__class__.__name__)
                #tmp_out={'models':[],'best_score_':[],'best_params_':[], 'test_r2':[], 'test_mean_squared_error':[]}
            if use_memory:
                cachedir = mkdtemp()
                memory = Memory(location=cachedir, verbose=False)
            else:     memory= None
                    
            #get estimator (pipeline) for the inner folds
            estimator=self.create_inner_estimator(modfinder,memory,prefix_dict=prefix_dict) 
                    
            #check given parameters, currently validates dimension dimension reduction methods
            par=self.check_parameters(par,cv,X_train.shape[0],X_train.shape[1],prefix_dict=prefix_dict)

            if self.verbose > 1: print(par)
                
            #search parameters and get best estimator
            paramsearch.search(estimator,par, scoring=scoring,cv=inner_group_kfold,X_train=X_train, y_train=y_train, random_state=random_state)
            if paramsearch.fit_params is not None and isinstance(paramsearch.get_best_estimator(), Pipeline):
                est=clone(paramsearch.get_best_estimator()).fit(X_train, y_train,**paramsearch.fit_params)
            else:
                est=clone(paramsearch.get_best_estimator()).fit(X_train, y_train)
            
            mn= self.model_to_string(est)
                    
            if mn in ests_list:
                if self.verbose > 1: print(f'the found model {mn} is not new and will not be added..')
                continue
            else:
                ests_list[mn]=1
                if self.verbose: print('new model added:',mn)
            
            if use_memory: rmtree(cachedir)
            
            #add scores using functionality from ModelFinder
            out=modfinder.add_inner_scores(out,est,X_test,y_test,paramsearch,mn)
        return out
    
    #implements outer fold parameter search using given ModelFinder and training data     
    def outer_fold_search(self,modfinder,X ,y,out,paramsearch,scoring,outer_group_kfold,cv,blender_params,use_memory,prefix_dict=None,nb_props=0,random_state=42):
        """
        find models using outer folds of nested cross-validation
        
        Args:
            modfinder: ModelFinder object to differentiate between classification and regression functionality such as scoring output
            X: data
            y: observations
            out: output dictionary
            paramsearch: a derived ParamSearch object that performs the parametersearch
            scoring: scorer string optimized in the parameter search
            outer_group_kfold: for outer folds
            cv: number of folds
            blender_params: list of parameters options
            use_memory: boolean to indidcate use of memory 
            prefix_dict: dictionary of prefixes 
            random_state: random state parameter
        
        Returns:
            dictionary with the top estimator in the key 'blender_model'
        """
        if use_memory:
            cachedir = mkdtemp()
            memory = Memory(location=cachedir, verbose=False)
        else:     memory= None
        
        #get estimator (stacking model or blender)
        stacked_model=self.create_outer_estimator(modfinder,memory,out,prefix_dict=prefix_dict,nb_props=nb_props) 

        #search parameters and get best estimator
        paramsearch.search(stacked_model,blender_params, scoring=scoring,cv=outer_group_kfold,X_train=X, y_train=y, random_state=random_state)
        if paramsearch.fit_params is not None and isinstance(paramsearch.get_best_estimator(), Pipeline):
            final_est=clone(paramsearch.get_best_estimator()).fit(X, y,**paramsearch.fit_params)
        else:
            final_est=clone(paramsearch.get_best_estimator()).fit(X, y)
        
        if not self.top_stacking:
            mn= self.model_to_string(final_est)
        else: 
            mn= self.stacked_model_to_string(final_est)
        
        if self.verbose: print('final model:',mn)
        out['blender_model']=final_est 
        
        if use_memory: rmtree(cachedir)

        return out
    
    def check_parameters(self,par,cv,rows,ncol,prefix_dict):
        """
        adjust the desired reduced_dimension based on the given dimensions of the training data
        
        Args:
            par: the parameter dictionary or list
            cv: the number of folds
            rows: the number of rows (samples) of the training data
            ncol: the number of columns (features) of the training dataz
            prefix_dict: the dictionary of prefixes used in the parameter grid
        
        Returns:
            updated parameter dictionary
        """
        dim_prefix=prefix_dict['dim_prefix']
        if not isinstance(par, dict):  #added since no access to internal hyperopt objects
            for p in par:
                r = re.compile(f"{dim_prefix}(.*)(n_components|k|max_features)")
                key_list=list(filter(r.match, p.keys()))
                for key in key_list:
                    if isinstance(p[key],list):
                        tmp=[]
                        for n in p[key]:
                            r=(cv-1)/cv
                            nrow= int(math.floor(rows * r))
                            #ncol=X_train.shape[1]
                            if n <  nrow and n < ncol: tmp.append(n)
                        if len(tmp) < len( p[key]):
                            p[key]=tmp
                            if self.verbose: print(f'n_components for dim red is adjusted to the min of dim of the data in the fold {nrow, ncol}:',p[key])
        return par
    
    def create_inner_estimator(self,modfinder,memory,prefix_dict=None):
        """
        returns estimator pipeline framework, actual methods are present in the parameter lists
        
        Args:
            modfinder: modelFinder object to retrieve stackingModel (regressor or classifier)
            memory: boolean indicating use of memory
            prefix_dict: dictionary with the method prefix in the pipeline
        
        Returns:
            pipeline of the estimator for the inner fold parameter optimization
        """
        return Pipeline([('normalizer', None),
                        (prefix_dict['dim_prefix'],'passthrough'),
                        (prefix_dict['method_prefix'], None)]   ,memory=memory ,verbose =False)
   
    def create_outer_estimator(self,modfinder,memory,out,prefix_dict=None,nb_props=0):
        """
        returns stacking estimator or pipeline with BaseEstimatorTransformer, actual methods are present in the parameter lists
        
        Args:
            modfinder: ModelFinder object to retrieve stacking model (regressor or classifier)
            memory: boolean indicating use of memory
            prefix_dict: dictionary with the method prefix in the pipeline
        
        Returns:
            pipeline of the estimator or stacking model for the outer fold parameter optimization
        """
        estimator_prefix=prefix_dict['estimator_prefix']
        #top level stacking model
        if self.top_stacking:
            blender= Pipeline([('normalizer', None),
                            (prefix_dict['dim_prefix'],'passthrough'),
                            (prefix_dict['method_prefix'], None)])
            estimator_l=[(f'{estimator_prefix}{index}',m) for index,m in enumerate(out['models'])]
            return modfinder.create_stacking_model(estimators=estimator_l, final_estimator=blender)  
        #return pipeline with BaseEstimatorTransformer the transformer containing the base estimators and the outputs from the base estimator are then given to 
        #the top level blender
        else:
            return Pipeline([('normalizer', None),
                            (prefix_dict['dim_prefix'],BaseEstimatorTransformer(out['models'],classification=modfinder.is_classifier(),nb_props=nb_props)),
                            (prefix_dict['method_prefix'], None)])
    
    def model_to_string(self,estimator):
        """
        function to format the estimator to string
        
        Args:
            estimator: given estimator
        
        Returns:
            string representation of the estimator
        """
        return ':'.join([str(s) for s in [estimator.steps[0][1],estimator.steps[1][1], estimator.steps[2][1]]])
    
    def stacked_model_to_string(self,stacked_estimator):
        """
        function to format the stacking model to string
        
        Args:
            stacked_estimator: given stacking model
        
        Returns:
            string representation of the stacking model
        """
        return 'Base estimators: '+','.join([self.model_to_string(estimator[1]) for estimator in stacked_estimator.estimators])+'  => blender: '+':'.join([str(s) for s in [stacked_estimator.final_estimator.steps[0][1],stacked_estimator.final_estimator.steps[1][1], stacked_estimator.final_estimator.steps[2][1]]])

#######################################################    
class NestedCVSingleModelSearch(NestedCVModelSearch): 
    '''
    NestedCVSingleStackSearch is a specialization of NestedCVBaseStackingSearch to find one stacking model using outer folds
    '''
    
    def inner_fold_search(self,modfinder,X_train ,y_train,X_test ,y_test,out,paramsearch, scoring,inner_group_kfold, cv, params_grid, ests_list, use_memory, prefix_dict=None, random_state=42):
        """
        empty method, no inner fold search in this class
        """
        return out
    
    #find one stacking model, small differences with the inner_fold_search from NestedCVBaseStackingSearch (scores, storages etc)
    def outer_fold_search(self,modfinder,X ,y,out,paramsearch,scoring,outer_group_kfold,cv,params_grid,use_memory,prefix_dict=None,nb_props=0,random_state=42):
        '''
        specialization of NestedCVModelSearch.outer_fold_search to find one model using the outer folds
        
        Args:
            modfinder: ModelFinder object to differentiate between classification and regression functionality such as scoring output
            X: data
            y: observations
            out: output dictionary
            paramsearch: a derived ParamSearch object that performs the parametersearch
            scoring: scorer string optimized in the parameter search
            outer_group_kfold: for outer folds
            cv: number of folds
            blender_params: list of parameters options
            use_memory: boolean to indidcate use of memory 
            prefix_dict: dictionary of prefixes 
            random_state: random state parameter
        
        Returns:  
            dictionary with the top estimator in the key 'blender_model'
        '''
        assert len(params_grid)==1, 'More than one method provided'
        if use_memory:
            cachedir = mkdtemp()
            memory = Memory(location=cachedir, verbose=False)
        else:     memory= None
        
        estimator=self.create_inner_estimator(modfinder,memory,prefix_dict=prefix_dict) 

        #check given parameters, currently validates dimension dimension reduction methods
        par=self.check_parameters(params_grid[0],cv,X.shape[0],X.shape[1],prefix_dict=prefix_dict)

        if self.verbose > 1: print(params_grid[0])

        #search parameters and get best estimator
        paramsearch.search(estimator,params_grid[0], scoring=scoring,cv=cv,X_train=X, y_train=y, random_state=random_state)
        if paramsearch.fit_params is not None and isinstance(paramsearch.get_best_estimator(), Pipeline):
            est=clone(paramsearch.get_best_estimator()).fit(X, y,**paramsearch.fit_params)
        else:
            est=clone(paramsearch.get_best_estimator()).fit(X, y)

        mn= self.model_to_string(est)
        if self.verbose > 1: print(f'Found model {mn}.')

        if use_memory: rmtree(cachedir)

        #add scores using functionality from ModelFinder
        out=modfinder.add_stacking_scores(out,est,paramsearch,scoring)
        out['blender_model']=est
        
        
        return out
    
###########################
class NestedCVBaseStackingSearch(NestedCVModelSearch):
    '''
    NestedCVBaseStackingSearch is a specialization of NestedCVModelSearch to find stacking models as base estimators using inner folds
    '''
    def __init__(self,verbose,top_stacking=False):
        """
        Initialization
        
        Args:
            verbose: boolean to set for print statements
            top_stacking: boolean to set to use a scikit stacking model for the top estimator
        """
        ## boolean or int to print things
        self.verbose=verbose
        ## boolean to indicate top_stacking
        self.top_stacking=top_stacking
        ## number of algorithms in stacking models, calculated based on given parameter dictionary
        self.nb_algos=None
    
    #inner fold search that finds one stacking model. (total number of stacking models equals the number of outer folds)
    def inner_fold_search(self,modfinder,X_train ,y_train,X_test ,y_test,out,paramsearch,scoring,inner_group_kfold, cv,params_grid,ests_list, use_memory, prefix_dict=None, random_state=42):
        '''
        specialization of NestedCVModelSearch.inner_fold_search to find stacking models instead of base estimators
        
        Args:
            modfinder: ModelFinder object to differentiate between classification and regression functionality such as scoring output
            X_train: training data
            y_train: training observations
            X_test: test data
            y_test: test observations
            out: output dictionary
            paramsearch: a derived ParamSearch object that performs the parametersearch
            scoring: scorer string optimized in the parameter search
            inner_group_kfold: inner folds
            cv: number of folds 
            params_grid: list of parameters options
            ests_list: list of already fitted/found base estimators
            use_memory: boolean to indidcate use of memory 
            prefix_dict: dictionary of prefixes 
            random_state: random state parameter 
        
        Returns:  
            dictionary with the found models inside key 'models'    
        """
        '''
        #get number of base estimators from first item of parameter list (each item is one stacking model)
        self.nb_algos=np.max([ int(re.search(r'\d+', key).group()) if (re.search(r'\d+', key) is not None) else 0 for key in params_grid[0].keys()])+1
        if self.verbose:
            print('****************')
            print('Searching models with the first params:',params_grid[0],' \n and more ... ')
        if use_memory:
            cachedir = mkdtemp()
            memory = Memory(location=cachedir, verbose=False)
        else:     memory= None  
        
        params_grid=self.check_parameters(params_grid,cv,X_train.shape[0],X_train.shape[1],prefix_dict=prefix_dict)     
        stacked_model=self.create_inner_estimator(modfinder,memory,prefix_dict=prefix_dict)  
        
        #search parameters and get best estimator
        paramsearch.search(stacked_model,params_grid, scoring=scoring,cv=inner_group_kfold,X_train=X_train, y_train=y_train, random_state=random_state)
        
        if paramsearch.fit_params is not None and isinstance(paramsearch.get_best_estimator(), Pipeline):
            est=clone(paramsearch.get_best_estimator()).fit(X_train, y_train,**paramsearch.fit_params)
        else:
            est=clone(paramsearch.get_best_estimator()).fit(X_train, y_train)
     
        mn= self.model_to_string(est)
        skip_model=False
        if mn in ests_list:
            skip_model=True
            if self.verbose > 1: print(f'the found model {mn} is not new and will not be added..')
        else:
            ests_list[mn]=1
            if self.verbose: print('new model added:',mn)
        
        out=modfinder.add_inner_scores(out,est,X_test,y_test,paramsearch,mn,skip_model)
        if use_memory: rmtree(cachedir)

        return out
        
    def check_parameters(self,par,cv,rows,ncol,prefix_dict=None):
        ''' 
        specialization of NestedCVModelSearch.check_parameters to verify parameters for stacking parameter list
        
        Args:
            par: the parameter dictionary
            cv: the number of folds
            rows: the number of rows (samples) of the training data
            ncol: the number of columns (features) of the training dataz
            prefix_dict: the dictionary of prefixes used in the parameter grid
        
        Returns:  
            updated parameter dictionary
        '''
        estimator_prefix=prefix_dict['estimator_prefix']
        dim_prefix=prefix_dict['dim_prefix']
        assert self.nb_algos is not None
        if not isinstance(par, dict):  #added since no access to internal hyperopt objects
            for p in par:
                r = re.compile(f"{estimator_prefix}(\b([0-9]|[1-9][0-9])\b)_{dim_prefix}(.*)(n_components|k|max_features)")
                key_list=list(filter(r.match, p.keys()))
                for key in key_list:
                    if isinstance(p[key],list):
                        tmp=[]
                        for n in p[key]:
                            r=(cv-1)/cv
                            nrow= int(math.floor(rows * r))
                            #ncol=X_train.shape[1]
                            if n <  nrow and n < ncol: tmp.append(n)
                        if len(tmp) < len( p[key]):
                            p[key]=tmp
                            if self.verbose: print(f'n_components for dim red is adjusted to the min of dim of the data in the fold {nrow, ncol}:',p[key])
        return par
    
    def create_inner_estimator(self,modfinder,memory,prefix_dict=None):
        ''' 
        specialization of NestedCVModelSearch.create_inner_estimator that returns stacking_model as inner fold estimator
        
        Args:
            modfinder: modelFinder object to retrieve stackingModel (regressor or classifier)
            memory: boolean indicating use of memory
            prefix_dict: dictionary with the method prefix in the pipeline
        
        Returns:  
            pipeline of the estimator for the inner fold parameter optimization
        '''
        estimator_prefix=prefix_dict['estimator_prefix']
        assert self.nb_algos is not None
        estimators=[  (f'{estimator_prefix}{index}',Pipeline([('normalizer', None),
                                                (prefix_dict['dim_prefix'],'passthrough'),
                                                (prefix_dict['method_prefix'], None)])) for index in range(self.nb_algos)]
        blender= Pipeline([('normalizer', None),
                            (prefix_dict['dim_prefix'],'passthrough'),
                            (prefix_dict['method_prefix'], None)])
        #return to ModelFinder for difference regression/stacking
        return modfinder.create_stacking_model(estimators=estimators, final_estimator=blender)                                             
    
    def model_to_string(self,stacked_estimator):
        """
        function to format the model to string
        
        Args:
            stacked_estimator: given stacking model
        
        Returns:  
            string representation of the stacking model
        """
        return ';'.join([':'.join([str(s) for s in [estimator[1].steps[0][1],estimator[1].steps[1][1], estimator[1].steps[2][1]]]) for estimator in stacked_estimator.estimators])+'  =>  '+':'.join([str(s) for s in [stacked_estimator.final_estimator.steps[0][1],stacked_estimator.final_estimator.steps[1][1], stacked_estimator.final_estimator.steps[2][1]]])

########################### 

class NestedCVSingleStackSearch(NestedCVBaseStackingSearch): 
    '''
    NestedCVSingleStackSearch is a specialization of NestedCVBaseStackingSearch to find one stacking model using outer folds
    '''
    
    def inner_fold_search(self,modfinder,X_train ,y_train,X_test ,y_test,out,paramsearch, scoring,inner_group_kfold, cv, params_grid, ests_list, use_memory, prefix_dict=None, random_state=42):
        """
        empty method, no inner fold search here
        """
        return out
    
    #find one stacking model, small differences with the inner_fold_search from NestedCVBaseStackingSearch (scores, storages etc)
    def outer_fold_search(self,modfinder,X ,y,out,paramsearch,scoring,outer_group_kfold,cv,params_grid,use_memory,prefix_dict=None,nb_props=0,random_state=42):
        '''
        specialization of NestedCVModelSearch.outer_fold_search to find one stacking model using the outer folds
        
        Args:
            modfinder: ModelFinder object to differentiate between classification and regression functionality such as scoring output
            X: data
            y: observations
            out: output dictionary
            paramsearch: a derived ParamSearch object that performs the parametersearch
            scoring: scorer string optimized in the parameter search
            outer_group_kfold: for outer folds
            cv: number of folds
            blender_params: list of parameters options
            use_memory: boolean to indidcate use of memory 
            prefix_dict: dictionary of prefixes 
            random_state: random state parameter
        
        Returns:  
            dictionary with the top estimator in the key 'blender_model'
        '''
        self.nb_algos=np.max([ int(re.search(r'\d+', key).group()) if (re.search(r'\d+', key) is not None) else 0 for key in params_grid[0].keys()])+1
        if self.verbose:
            print('****************')
            print('Searching models with the first params:',params_grid[0],' \n and more ... ')
        if use_memory:
            cachedir = mkdtemp()
            memory = Memory(location=cachedir, verbose=False)
        else:     memory= None  
            
        params_grid=self.check_parameters(params_grid,cv,X.shape[0],X.shape[1],prefix_dict=prefix_dict)
        #same config, reuse inner_estimator from NestedCVBaseStackingSearch
        stacked_model=self.create_inner_estimator(modfinder,memory,prefix_dict=prefix_dict)
        
        #search parameters and get best estimator
        paramsearch.search(stacked_model,params_grid, scoring=scoring,cv=outer_group_kfold,X_train=X, y_train=y, random_state=random_state)
        if paramsearch.fit_params is not None and isinstance(paramsearch.get_best_estimator(), Pipeline):
            est=clone(paramsearch.get_best_estimator()).fit(X, y,**paramsearch.fit_params)
        else:
            est=clone(paramsearch.get_best_estimator()).fit(X, y)

        mn= self.model_to_string(est)
        
        if self.verbose: print('model:',mn)
        
        out=modfinder.add_stacking_scores(out,est,paramsearch,scoring)

        if use_memory: rmtree(cachedir)
        
        out['blender_model']=est
        
        return out
######################################################
#import pickle
#import dill

class ModelFinder:
    '''
    ModelFinder is called from FeatureGenerationRegressor to search for the best parameters using nested model selection by calling functionality from NestedCVModelSearch.
    '''
    def __init__(self,verbose,groupfold,outer_jobs=None):
        """
        Initialization
        
        Args:
            verbose: boolean/int to print statements
            groupfold: NestedCVModelSearch object to call inner and outer fold searches
            outer_jobs: number of threads used to perform outer fold cross-validation
        """
        ## enable print statements
        self.verbose=verbose
        ## object of NestedCVModelSearch
        self.groupfold=groupfold
        ## scoring list
        self.scorings=[]
        ## number of threads
        self.outer_jobs=outer_jobs
        
        
    def nested_fold_i(self, arg,cache, proxy, lock,X, y, sample_weight, paramsearch, groups, scoring, params_grid, cv,use_memory,prefix_dict,random_state,split):
        """
        a function that is called in using multiprocessing and performans the inner-cross validation in parallel
        
        Args:
            arg: provided arguments from nested multiprocessing
            cache: provided cache from nested multiprocessing
            lock: provided lock from nested multiprocessing
            X: Data matrix X
            y: observations
            sample_weight: sample weight as array
            paramsearch: ParamSearch object performing parameter optimization
            groups: non overlapping clusters 
            scoring: scoring function as string
            params_grid: parameter grid to be searched
            cv: number of inner folds
            use_memory: boolean to indicate memory use
            prefict_dict: dictionary of prefixes
            random-state: random state integer
            split: string representing the cross-validation split
               LGO = LeaveOneGroupOut CV
               GKF = Using GroupKFold CV
               SKF = Stratified k-fold CV 
        
        Returns:  
            dictionary with the results
        """
        i, (train_index, test_index) = arg
        out=self.initialise_output()
        if self.verbose: print('**********************************************\n',f'---- outer CV {i+1} ----' )
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if sample_weight is not None:
            est_prefix=prefix_dict['method_prefix']
            paramsearch.add_attribute_fit_params(key=f'{est_prefix}__sample_weight', value=sample_weight[train_index] )

        groups_train= groups[train_index]
        inner_group_kfold,_=self.groupfold.get_folds(X_train ,y_train,groups_train,cv,split)
        #search parameters using inner cross-validation for this outer-fold
        out=self.groupfold.inner_fold_search(self,X_train ,y_train,X_test ,y_test,out,paramsearch, scoring,list(inner_group_kfold), cv,params_grid,{}, use_memory, prefix_dict=prefix_dict, random_state=random_state )

        return out
    
    #performs model selection by calling inner and outer fold functionality from the specific NestedCVModelSearch using the given groups  
    def model_search(self,X ,y,X_blender=None, groups=None,params_grid=None, paramsearch=None, scoring='r2' ,cv=3
                                ,use_memory=True,verbose=1 , outer_cv_fold=5
                                ,split='SKF',blender_params=None,prefix_dict=None,random_state=42,sample_weight=None):
        """
        Nested cross-validation to find models by calling inner and outer fold search from NestedCVModelSearch
        
        Args:
            X: data
            y: observations
            groups: non-overlapping data sample groups
            params_grid: list of parameters options for base estimators
            paramsearch: Paramsearch object to search the parameters
            scoring: scorer string optimized in the parameter search
            cv: number of inner folds
            use_memory: boolean to indidcate use of memory 
            verbose: boolean to print statements
            outer_cv_fold: number of outer folds
            split: string to indicate which splitting method to be used.
               LGO = LeaveOneGroupOut CV
               GKF = Using GroupKFold CV
               SKF = Stratified k-fold CV
            blender_params: list of parameters options for top estimators
            prefix_dict: dictionary of prefixes 
            random_state: random state parameter 
            sample_weight: sample weight as array
        
        Returns:  
            dictionary with the result and models
        """
        assert params_grid is not None, 'provide grid parameters'
        #remove fit params
        paramsearch.clear_fit_params()
        
        out=self.initialise_output()
        if self.verbose: print(f' Performing nested CV with {outer_cv_fold} outer loop and {cv} inner loop')
        outer_group_kfold,nb_data_splits=self.groupfold.get_folds(X ,y,groups,outer_cv_fold,split)
        i=0
        #list of found estimators
        ests_list={}

        for train_index, test_index in outer_group_kfold:
            if self.verbose: print('$$$$$$$$$$$$$$$$$$$$\n',f'---- outer CV {i+1} ----' )
            start_time = time.time()
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            groups_train= groups[train_index]
            if sample_weight is not None:
                est_prefix=prefix_dict['method_prefix']
                paramsearch.add_attribute_fit_params(key=f'{est_prefix}__sample_weight', value=sample_weight[train_index] )

            inner_group_kfold,_=self.groupfold.get_folds(X_train ,y_train,groups_train,cv,split)

            #search parameters using inner cross-validation for this outer-fold
            out=self.groupfold.inner_fold_search(self,X_train ,y_train,X_test ,y_test,out,paramsearch, scoring,list(inner_group_kfold), cv,params_grid,ests_list, use_memory, prefix_dict=prefix_dict, random_state=random_state )

            i+=1
            if self.verbose: 
                minutes, seconds = divmod(time.time() - start_time, 60)
                print(f'performed outer CV {i} in {minutes} min and {seconds} seconds' )
        if blender_params is not None:
            if sample_weight is not None:
                est_prefix=prefix_dict['method_prefix']
                red_prefix=prefix_dict['dim_prefix']
                paramsearch.add_attribute_fit_params(key=f'{red_prefix}__{est_prefix}__sample_weight', value=sample_weight )
                paramsearch.add_attribute_fit_params(key=f'{est_prefix}__sample_weight', value=sample_weight)
            #find parameters top level estimator/blender
            outer_group_kfold,_=self.groupfold.get_folds(X ,y,groups,outer_cv_fold,split)
            #do outer fold search, can be overwritten by inheritance
            start_time = time.time()
            if self.verbose: print('Fitting blender, final_estimator using cross-validation' )
            if X_blender is not None:
                X_b=np.hstack((X,X_blender))
                nb_props=X_blender.shape[1]
            else:
                X_b=X
                nb_props=0
            out=self.groupfold.outer_fold_search(self,X_b ,y, out, paramsearch, scoring, list(outer_group_kfold), cv, blender_params, use_memory, prefix_dict=prefix_dict,nb_props=nb_props, random_state=random_state)
            if self.verbose:
                minutes, seconds = divmod(time.time() - start_time, 60)
                print(f'performed outer CV final estimator fit in {minutes} min and {seconds} seconds' ) 
             
        out=self.add_outer_scores(out)

        return out
    
    def process_parallel_output_list(self,out_list):
        """
        processes the output of the multiprocessing and stores only unique methods
        
        Args:
            out_list: list of output dictionaries
        
        Returns:  
            merged output dictionary
        """
        out=out_list[0]
        #print(out_list)
        if self.verbose: 
            for mn in out['model_str']:
                print('new model added:',mn)
        for out_i in out_list[1:]:
            #check uniqueness models from parallel list
            for est,mn in zip(out_i['models'],out_i['model_str']):
                if mn in out['model_str']:
                    if self.verbose > 1: print(f'the found model {mn} is not new and will not be added..')
                else:
                    out['model_str'].append(mn)
                    out['models'].append(est)
                    if self.verbose: print('new model added:',mn)
            del out_i['model_str']
            del out_i['models']
            for key,item in out_i.items():
                out[key].extend(item)
        return out
        
    
    def is_classifier(self):
        """
        returns True if classifier
        
        Returns:  
            False
        """
        return False
    
    #functionality to get mean and std specifically for stacking model
    def add_stacking_scores(self,out,est,grid,scoring):
        """
        function to add scores from stacking model
        
        Args:
            out: output dictionary
            est: stacking model
            grid: paramsearch object with attribute best_params_ and best_index_
            scoring: scoring function represented as string
        
        Returns:  
            updated output dictionary with keys: models, best_params_, test_<scoring>
        """
        #standard scores not available
        self.scorings=[scoring]
        out['models'].append(est)
        out['best_params_'].append(grid.best_params_)
        out[f'test_{scoring}']=[]
        nb_folds=np.max([ int(re.search('split(\d+)', key).group(1)) if (re.search('split(\d+)', key) is not None) else 0 for key in sorted(grid.cv_results_)])+1
        for fold in range(nb_folds):
            out[f'test_{scoring}'].append(grid.cv_results_[f'split{fold}_test_score'][grid.best_index_])
        return out
        
    def create_stacking_model(self,estimators=None, final_estimator=None):
        """
        empty method
        """
        pass
    
    #add in specializations
    def initialise_output(self):
        """
        empty method
        """
        pass
        
    def add_inner_scoring(self,out,est,X_test,y_test,grid):
        """
        empty method
        """
        pass
        
    def add_outer_scores(self,out):
        """
        empty method
        """
        pass

###########################
class ClassificationFinder(ModelFinder):
    '''
    ClassificationFinder is specialization of ModelFinder for classification models.
    '''
    def __init__(self,verbose,groupfold,outer_jobs=None):
        """
        Initialization
        
        Args:
            verbose: boolean/int to print statements
            groupfold: NestedCVModelSearch object to call inner and outer fold searches
            outer_jobs: number of threads used to perform outer fold cross-validation
        """
        super().__init__(verbose,groupfold,outer_jobs)
        self.scorings = ['acc', 'auc']
    
    def create_stacking_model(self,estimators=None, final_estimator=None):
        """
        creates and returns StackingClassifier
        
        Args:
            estimators: list of base estimators
            final_estimator: final estimator of the stacking model
        
        Returns:  
            scikit StackingClassifier
        """
        return StackingClassifier(estimators=estimators, final_estimator=final_estimator)

        
    def initialise_output(self):
        """
        initialize classification output
        
        Returns:  
            dictionary with classification keys
        """
        return {'models':[],'model_str':[],'best_score_':[],'best_params_':[], 'test_acc':[], 'test_auc':[]}
  
    def add_inner_scores(self,out,est,X_test,y_test,grid,mn,skip_model=False):
        """
        saves classification inner cross-validation scores and models to output dictionary
        
        Args:
            out: output dictionary
            est: estimator
            X_test: data matrix for which scores are calculated
            y_test: true values
            grid: ParamSearch object
            mn: string of model
            skip_model: boolean to indicate that the model is not added to the list of models
        
        Returns:  
            updated output dictionary
        """
        y_pred_proba=est.predict_proba(X_test)
        try:
            score = est.score(X_test, y_test)
        except ValueError as ve:
            score=0.0
        if not skip_model:
            out['models'].append(est)
            out['model_str'].append(mn)
        out['best_params_'].append(grid.best_params_)
        out['test_acc'].append(score)
        if isinstance(y_pred_proba,list):
            for i,y_proba in enumerate(y_pred_proba):
                if y_proba.shape[1]>2:
                    out['test_auc'].append(roc_auc_score(y_test[:,i], y_proba,multi_class='ovo'))
                else:
                    out['test_auc'].append(roc_auc_score(y_test[:,i]==1, y_proba[:,1]))
        else:
            if y_pred_proba.shape[1]>2:
                out['test_auc'].append(roc_auc_score(y_test, y_pred_proba,multi_class='ovo'))
            else:
                out['test_auc'].append(roc_auc_score(y_test==1, y_pred_proba[:,1]))
        return out
    
    def is_classifier(self):
        """
        returns True for classification
        
        Returns:  
            True
        """
        return True
    
    def add_outer_scores(self,out):
        """
        saves classification outer cross-validation scores to out
        
        Args:
            out: output dictionary
        
        Returns:  
            updated output dictionary with key Nested_CV score and meand plus std of the scoring function on the test values
        """
        for scoring in self.scorings:
            m=np.mean(out[f'test_{scoring}'])
            sd=np.std(out[f'test_{scoring}'])
            if scoring =='mean_squared_error':
                scoring='MSE'
                m=np.abs(m)
            out[f'mean_test_{scoring}']=m
            out[f'std_test_{scoring}']=sd
        out['Nested_CV score']='Nested CV scores:'
        for scoring in self.scorings:
            if scoring =='mean_squared_error':scoring='MSE'
            out['Nested_CV score']+=' ' + '{}={:.2f} +/-{:.2f}'.format(scoring ,out[f'mean_test_{scoring}'],out[f'std_test_{scoring}'])
        if self.verbose:
            print(out['Nested_CV score'])
            print('Number of found models:', len(out['models']))
        return out


###########################
class RegressionFinder(ModelFinder):
    '''
    RegressionFinder is specialization of ModelFinder for regression models.
    '''
    def __init__(self,verbose,groupfold,outer_jobs=None):
        """
        Initialization
        
        Args:
            verbose: boolean/int to print statements
            groupfold: NestedCVModelSearch object to call inner and outer fold searches
            outer_jobs: number of threads used to perform outer fold cross-validation
        """
        super().__init__(verbose,groupfold,outer_jobs)
        self.scorings = ['r2', 'mean_squared_error']
        
    def initialise_output(self):
        """
        initialize regression output
        
        Returns:  
            dictionary with regression keys
        """
        return {'models':[],'model_str':[],'best_score_':[],'best_params_':[], 'test_r2':[], 'test_mean_squared_error':[]}

    def create_stacking_model(self,estimators=None, final_estimator=None):
        """
        creates and returns StackingRegressor
        
        Args:
            estimators: list of base estimators
            final_estimator: final estimator of the stacking model
        
        Returns:  
            scikit StackingRegressor
        """
        return StackingRegressor(estimators=estimators, final_estimator=final_estimator)
    
    def add_inner_scores(self,out,est,X_test,y_test,grid,mn,skip_model=False):
        """
        saves regression inner cross-validation scores and models to output dictionary
        
        Args:
            out: output dictionary
            est: estimator
            X_test: data matrix for which scores are calculated
            y_test: true values
            grid: ParamSearch object
            mn: string of model
            skip_model: boolean to indicate that the model is not added to the list of models
        
        Returns:  
            updated output dictionary
        """
        y_pred=est.predict(X_test)
        if not skip_model:
            out['models'].append(est)
            out['model_str'].append(mn)
        out['best_params_'].append(grid.best_params_)
        out['test_r2'].append(r2_score(y_test, y_pred))
        out['test_mean_squared_error'].append(mean_squared_error(y_test, y_pred))
        return out
        
    def add_outer_scores(self,out):
        """
        saves regression outer cross-validation scores to out
        
        Args:
            out: output dictionary
        
        Returns:  
            updated output dictionary with key Nested_CV score and meand plus std of the scoring function on the test values
        """
        for scoring in self.scorings:
            m=np.mean(out[f'test_{scoring}'])
            sd=np.std(out[f'test_{scoring}'])
            if scoring =='mean_squared_error':
                scoring='MSE'
                m=np.abs(m)
            out[f'mean_test_{scoring}']=m
            out[f'std_test_{scoring}']=sd
        out['Nested_CV score']='Nested CV scores:'
        for scoring in self.scorings:
            if scoring =='mean_squared_error':scoring='MSE'
            out['Nested_CV score']+=' ' + '{}={:.2f} +/-{:.2f}'.format(scoring ,out[f'mean_test_{scoring}'],out[f'std_test_{scoring}'])
        if self.verbose:
            print(out['Nested_CV score'])
            print('Number of found models:', len(out['models']))
        return out
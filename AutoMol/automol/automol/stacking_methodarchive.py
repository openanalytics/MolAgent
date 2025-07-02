"""implementation of the method dictionaries and their default hyperparameter options for the different cases.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""
 
import numpy as np

import sklearn
from sklearn.linear_model import LogisticRegression , Ridge, Lasso, RidgeClassifier, SGDClassifier, HuberRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, RationalQuadratic, ExpSineSquared,DotProduct, WhiteKernel ,Matern
from sklearn.feature_selection import SelectKBest,f_regression ,SelectPercentile,mutual_info_regression, VarianceThreshold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, BayesianRidge,SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif,f_regression, mutual_info_regression
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.neighbors import NeighborhoodComponentsAnalysis

import scipy.stats as stats

from packaging import version
if version.parse(sklearn.__version__)>=version.parse("1.2"):
    from scipy.stats import loguniform
else:
    from sklearn.utils.fixes import loguniform
from .mlpwrappers import MLPRegressorWrapper, MLPClassifierWrapper

import xgboost as xgb

from lightgbm import LGBMClassifier,LGBMRegressor

import hyperopt
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials


######################################################
class MethodArchive:
    """
    Base Class to store and alter the different methods using the same basic functionality
    """

    def add_method(self,name,method,param_dictionary):
        """
        Add a new method to the archive defined by its name in the dictionary, the class with the scikit estimator functionality and the parameters.
        
        The method_prefix for the parameters is checked and can thus be added or omitted.
        
        In the case of hyperopt optimization the parameter choices have to hyperopt objects (eg. hyperopt.choice or ...)
        
        Args:
             name: the key used in the underlying dictionary, choose something appropriate
             method: the method class
             param_dictionary: the different parameters as a dictionary
        """
        if self.hyperopt_defaults:
            self.methods[name]={**{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_{name}',[method])}, **{ f'{keys}' if (keys.startswith(f'{self.method_prefix}__')) else f'{self.method_prefix}__{keys}': hp.choice(f'{keys}_{name}',values) if (keys.startswith(f'{self.method_prefix}__')) else hp.choice(f'{self.method_prefix}__{keys}_{name}',values) for keys,values in param_dictionary.items() } }
        else:
            self.methods[name]={**{f'{self.method_prefix}':[method]}, **{ f'{keys}' if (keys.startswith(f'{self.method_prefix}__')) else f'{self.method_prefix}__{keys}':values for keys,values in param_dictionary.items() } }
        
    
    def add_param(self,name,param_name,param_val):
        """
        add or change the parameter for a method defined by its name in the dictionary, the name of the parameter in the method dictionary and the value
        
        the param_name can be with of without method prefix. In the case of hyperopt optimization param_val has to be a hyperopt object (eg. hyperopt.choice or ...)
        
        Args:
             name: the key used in the estimator/method dictionary
             param_name: they key of the parameter in the parameter dictionary, this is the name of the parameter of the estimator
             param_val: the values/distribution of the given parameter. For hyperopt this must be hyperopt.choice object or another hyperopt object
        """
        assert name in self.methods
        if param_name.startswith(f'{self.method_prefix}__'):
            assert hasattr(self.methods[name][self.method_prefix][0], param_name.replace(f'{self.method_prefix}__','')) or (name.startswith('xgb') and param_name.replace(f'{self.method_prefix}__','')=='random_state'), f'param name {param_name} is not present'
            self.methods[name][param_name]=param_val
        else:
            assert hasattr(self.methods[name][self.method_prefix][0], param_name) or (name.startswith('xgb') and param_name=='random_state'), f'param name {param_name} is not present'
            self.methods[name][f'{self.method_prefix}__{param_name}']=param_val
            
    def duplicate_method_xtimes(self,method_name,x,random_state=None):
        """
        duplicates a method x times with different random_states initializations and returns the corresponding dictionary keys
        
        Args:
             method_name: the dictionary key of the existing estimator
             x: number of copies
             random_state: random state initialisation, if value and not a list, a list is generated from this value
        """
        
        assert not self.hyperopt_defaults, 'duplicating methods with hyperopt is not supported, work around given in stacking_util for xgb and lgbm'
       
        if random_state is not None and not isinstance(random_state, (list)):
            random_state=[(random_state+(7+random_state)**i)%31393 for i in range(x)]
        if random_state is None:
            random_state=[None for i in range(x)]
        assert len(random_state)==x, 'len of random states is not equal to number of method duplicates'
        original=self.get_method(method_name).copy()
        est=original[self.method_prefix][0]
        del original[self.method_prefix]
        est_params=original
        clf_list=[]
        for copy_i in range(x):
            m_name=f'{method_name}_{copy_i}'
            clf_list.append(m_name)
            self.add_method(m_name,est,est_params)
            self.add_param(m_name,'random_state',[random_state[copy_i]])
        return clf_list
    
    def get_method(self,name):
        """
        returns the method dictionary entry (with parameters) corresponding to the given key
        
        Args:
             name: estimator key
        
        Returns: 
            method corresponding to given key
        """
        assert name in self.methods, 'name not in method dictionary, use add_method to add the method corresponding to the given name. Verify the available method by using the function self.get_all_method_keys()'
        return self.methods[name]
    
    def get_all_method_keys(self):
        """
        returns all keys
        
        Returns: 
            all dictionary keys in the archive
        """
        return self.methods.keys()
    
    def get_all_method_keys_plus_estimators(self):
        """
        returns list of tuples with keys and estimator classes
        
        Returns: 
            keys and estimators as tuple
        """
        return [(key,self.methods[key][self.method_prefix]) for key in self.methods.keys()]
    
    def get_methods(self,names):
        """
        returns list of method dictionaries based on the given list of estimator keys
        
        Returns: 
            list of method dictionaries corresponding to the given keys
        """
        return [self.get_method(name) for name in names]
###########################
class RegressorArchive(MethodArchive):
    """
    derived class of MethodArchive with the defaults regressor estimators and their parameters
    """
    def __init__(self,
                 method_prefix='reg',
                 distribution_defaults=False,
                 hyperopt_defaults=False,
                 xgb_threads=4,
                 rfr_threads=-1,
                 method_jobs=None,
                 n_estimators=None,
                 pls_n_components=None,
                 gp_kernels=None,
                random_state=42):
        """
        Initialization
        
        Args:
             method_prefix: the method prefix
             distribution_defaults: boolean for default distributional parameters
             hyperopt_defaults: boolean to be set for default parameters for HyperOpt optimization
             xgb_threads: number of threads for XGBoost
             rfr_threads: number of threads for RandomForest
             method_jobs: number of threads for general methods
             n_estimators: number of estimators for tree based ensembles (Random forest, XGboost, Adaboost)
             pls_n_components: number of components for pls
             gp_kernels: gaussian process kernels
             random_state: the random state init value
        """
        self.method_prefix=method_prefix
        if pls_n_components is None: pls_n_components=[1,2,5,10,20,30,40,100,200]
        if n_estimators is None : n_estimators=[100,250,500,1000]
        if gp_kernels is None: gp_kernels=[None, DotProduct() + WhiteKernel(),Matern()]
        if isinstance(random_state, (list)):
            random_state_list=random_state
        else:
            random_state_list=[random_state]
        self.hyperopt_defaults=hyperopt_defaults
        
        if method_jobs is None:
            if xgb_threads is not None:
                method_jobs = xgb_threads
            elif rfr_threads is not None:
                method_jobs=rfr_threads
            else:
                method_jobs=2
                rfr_threads=method_jobs
                xgb_threads=method_jobs
        if rfr_threads is None:
            rfr_threads=method_jobs
        if xgb_threads is None:
            xgb_threads=method_jobs
            
        
        if distribution_defaults:
            if hyperopt_defaults:
                self.methods={
                    'lasso':{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_lasso',[Lasso(tol=1e-2,max_iter=500)]),
                             f'{self.method_prefix}__alpha':hp.loguniform(f'{self.method_prefix}__alpha_lasso',-5, 2),
                             f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_lasso',random_state_list)},
                    'huber':{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_huber',[HuberRegressor(tol=1e-2,max_iter=500)]),
                             f'{self.method_prefix}__alpha':hp.loguniform(f'{self.method_prefix}__alpha_huber',-5, 2),
                             f'{self.method_prefix}__epsilon':hp.uniform(f'{self.method_prefix}__epsilon_huber',1.0,1.7)},
                    'pls' : {f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_pls',[PLSRegression(tol=1e-3,max_iter=500)]),
                             f'{self.method_prefix}__n_components':hp.choice(f'{self.method_prefix}__n_components_pls',pls_n_components)  },
                    'xgb' :{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_xgb',[xgb.XGBRegressor(objective ='reg:squarederror',verbosity =1,nthread=xgb_threads)]),
                              f'{self.method_prefix}__eval_metric':hp.choice(f'{method_prefix}__eval_metric_xgb', ['rmse','mae','logloss']),
                              f'{self.method_prefix}__n_estimators': hp.choice(f'{self.method_prefix}__n_estimators_xgb',n_estimators),
                              f'{self.method_prefix}__max_depth': hp.choice(f'{self.method_prefix}__max_depth_xgb',range(3,8)),
                             f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_xgb',random_state_list),
                              f'{self.method_prefix}__min_child_weight': hp.quniform(f'{self.method_prefix}__min_child_weight_xgb',2,14,2),
                              f'{self.method_prefix}__gamma': hp.quniform(f'{self.method_prefix}__gamma_xgb',0.5, 5,0.5),
                              f'{self.method_prefix}__learning_rate': hp.loguniform(f'{self.method_prefix}__learning_rate_xgb',-4, -1),
                              f'{self.method_prefix}__subsample': hp.uniform(f'{self.method_prefix}__subsample_xgb',0.5,1.0),
                              f'{self.method_prefix}__colsample_bytree': hp.uniform(f'{self.method_prefix}__colsample_bytree_xgb',0.6,1.0)
                            },
                    'lgbm' :{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_lgbm',[LGBMRegressor(n_jobs=xgb_threads,verbosity=-1)]),
                              f'{self.method_prefix}__n_estimators':hp.choice(f'{self.method_prefix}_lgbm__n_estimators', n_estimators),
                              f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}_lgbm__random_state',random_state_list),
                              f'{self.method_prefix}__max_depth': hp.choice(f'{self.method_prefix}_lgbm__max_depth',range(3,8)),
                              f'{self.method_prefix}__reg_alpha': hp.loguniform(f'{self.method_prefix}_lgbm__reg_alpha',-4, 1),
                              f'{self.method_prefix}__reg_lambda': hp.loguniform(f'{self.method_prefix}_lgbm__reg_lambda',-4, 1),
                              f'{self.method_prefix}__min_child_samples': hp.choice(f'{self.method_prefix}_lgbm__min_child_samples',range(2, 14, 3)),
                              f'{self.method_prefix}__subsample': hp.choice(f'{self.method_prefix}_lgbm__subsample',[0.5,0.8,1.0]),
                              f'{self.method_prefix}__learning_rate': hp.loguniform(f'{self.method_prefix}_lgbm__learning_rate',-4, -1),
                              f'{self.method_prefix}__num_leaves': hp.choice(f'{self.method_prefix}_lgbm__num_leaves',[2,5,10,15,20,30]),
                              f'{self.method_prefix}__boosting_type':hp.choice(f'{self.method_prefix}_lgbm__boosting_type',['gbdt','dart'])
                            },
                    'mlp' :{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_mlp',[MLPRegressorWrapper()]),
                              f'{self.method_prefix}__hidden_layers':hp.choice(f'{self.method_prefix}_mlp__hidden_layers', range(1,5)),
                              f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}_mlp__random_state',random_state_list),
                              f'{self.method_prefix}__hidden_layers_size': hp.choice(f'{self.method_prefix}_mlp__hidden_layers_size',[10,20,50,100]),
                              f'{self.method_prefix}__learning_rate_init': hp.loguniform(f'{self.method_prefix}_mlp__learning_rate_init',-5, -1),
                              f'{self.method_prefix}__max_iter': hp.choice(f'{self.method_prefix}_mlp__max_iter',[50,100,200,400])
                            },
                    'svr': {f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_svr',[SVR(tol=1e-2,max_iter=500)]),
                            f'{self.method_prefix}__C':  hp.loguniform(f'{self.method_prefix}__C_svr',-3, 3),
                            f'{self.method_prefix}__epsilon':  hp.loguniform(f'{self.method_prefix}__epsilon_svr',-3, 3),
                            f'{self.method_prefix}__gamma': hp.loguniform(f'{self.method_prefix}__gamma',-3, 3)
                            #,f'{self.method_prefix}__kernel': ['linear', 'rbf']
                           },
                   # 'rfr' : { f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_rfr',[RandomForestRegressor(n_jobs=rfr_threads)]),
                   #           f'{self.method_prefix}__criterion':hp.choice(f'{self.method_prefix}__criterion_rfr',['squared_error','poisson']),
                   #          f'{self.method_prefix}__n_estimators' : hp.choice(f'{self.method_prefix}__n_estimators_rfr',n_estimators),
                   #          f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_rfr',random_state_list)
                   #          #,f'{self.method_prefix}__max_features' : ["auto", "sqrt", "log2"]
                   #          ,f'{self.method_prefix}__max_depth' : hp.choice(f'{self.method_prefix}__max_depth_rfr',[3,4,5,8])
                   #          ,f'{self.method_prefix}__min_samples_split' : hp.choice(f'{self.method_prefix}__min_samples_split_rfr',[2,4,8])
                   #          ,f'{self.method_prefix}__min_samples_leaf':hp.choice(f'{self.method_prefix}__min_samples_leaf_rfr',np.logspace(-4, -1, 8)) 
                   #          #,f'{self.method_prefix}__bootstrap': [True, False]
                   #         },
                   # 'ada':{ f'{self.method_prefix}':hp.choice(f'{self.method_prefix}_ada', [AdaBoostRegressor()]),
                   #          f'{self.method_prefix}__n_estimators' :hp.choice(f'{self.method_prefix}__n_estimators_ada', n_estimators),
                   #          f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_ada',random_state_list)
                   #          ,f'{self.method_prefix}__learning_rate' : hp.choice(f'{self.method_prefix}__learning_rate_ada',np.logspace(-3, 2, 10))
                   #         },
                    'bayesianridge': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}__br',[BayesianRidge()]),
                                   f'{self.method_prefix}__alpha_1':hp.loguniform(f'{self.method_prefix}__alpha_1_rb',-8, -1),
                                    f'{self.method_prefix}__alpha_2':hp.loguniform(f'{self.method_prefix}__alpha_2_br',-8, -1),
                                    f'{self.method_prefix}__lambda_1':hp.loguniform(f'{self.method_prefix}__lambda_1_br',-8, -1),
                                    f'{self.method_prefix}__lambda_2':hp.loguniform(f'{self.method_prefix}__lambda_2_br',-8, -1)
                                  },
                    'dtr': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}__dtr',[DecisionTreeRegressor()]),
                              f'{self.method_prefix}__criterion':hp.choice(f'{self.method_prefix}__criterion_dtr',['squared_error']),
                              f'{self.method_prefix}__max_depth':hp.choice(f'{self.method_prefix}__max_depth_dtr',range(3,8)),
                              f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_dtr',random_state_list),
                              f'{self.method_prefix}__min_samples_split': hp.choice(f'{self.method_prefix}__min_samples_split_dtr',range(2,10,2)),
                              f'{self.method_prefix}__min_samples_leaf':hp.loguniform(f'{self.method_prefix}__min_samples_leaf_dtr',-4, -1)  
                           },
                    'sgdr': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_sgdr',[SGDRegressor()]),
                                   f'{self.method_prefix}__alpha':hp.loguniform(f'{self.method_prefix}__alpha_sgdr',-6,2),
                                   f'{self.method_prefix}__max_iter':hp.choice(f'{self.method_prefix}__max_iter_sgdr',[100,500,1000,3000]),
                                   f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_sgdr',random_state_list),
                                   f'{self.method_prefix}__tol': hp.loguniform(f'{self.method_prefix}__tol_sgdr',-6, -2),
                                   f'{self.method_prefix}__penalty': hp.choice(f'{self.method_prefix}__penalty_sgdr',['l2','l1','elasticnet'])
                                  },
                    'kernelridge': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_kernelridge',[KernelRidge(kernel='rbf')]),
                                   f'{self.method_prefix}__alpha':hp.loguniform(f'{self.method_prefix}__alpha_kernelridge',-4, 2),
                                   f'{self.method_prefix}__gamma': hp.loguniform(f'{self.method_prefix}__gamma_kernelridge',-3, 1 )
                                  },
                    'gp': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_gp',[GaussianProcessRegressor()]),
                                f'{self.method_prefix}__kernel':hp.choice(f'{self.method_prefix}__kernel_gp',gp_kernels),
                                f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_gp',random_state_list)
                            },
                }
            else:
                self.methods={
                    'lasso':{f'{self.method_prefix}': [Lasso(tol=1e-2,max_iter=500)],
                             f'{self.method_prefix}__alpha':loguniform(1e-5,1e2),
                             f'{self.method_prefix}__random_state':random_state_list },
                    'huber':{f'{self.method_prefix}': [HuberRegressor(tol=1e-2,max_iter=500)],
                             f'{self.method_prefix}__alpha':loguniform(1e-5,1e2),
                             f'{self.method_prefix}__epsilon':[1.0,1.15,1.3,1.35,1.5,1.7]},
                    'pls' : {f'{self.method_prefix}': [PLSRegression(tol=1e-3,max_iter=500)],
                             f'{self.method_prefix}__n_components':pls_n_components  },
                    'xgb' :{f'{self.method_prefix}': [xgb.XGBRegressor(objective ='reg:squarederror',verbosity =1,nthread=xgb_threads)],
                              f'{self.method_prefix}__eval_metric':['rmse','mae','logloss'],
                              f'{self.method_prefix}__n_estimators': n_estimators,
                              f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__max_depth': range(3,8),
                              f'{self.method_prefix}__min_child_weight': range(2, 14, 3),
                              f'{self.method_prefix}__gamma': [0.5, 1, 1.5, 2, 5],
                              f'{self.method_prefix}__learning_rate': loguniform(1e-3,2e-1),
                              f'{self.method_prefix}__subsample': [0.5,0.8,1.0],
                              f'{self.method_prefix}__colsample_bytree': [0.6,0.8,1.0]
                            },
                    'lgbm' :{f'{self.method_prefix}': [LGBMRegressor(n_jobs=xgb_threads,verbosity=-1)],
                              f'{self.method_prefix}__n_estimators': n_estimators,
                              f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__max_depth': range(3,8),
                              f'{self.method_prefix}__reg_alpha': loguniform(1e-4, 1e1),
                              f'{self.method_prefix}__reg_lambda': loguniform(1e-4, 1e1),
                              f'{self.method_prefix}__min_child_samples': range(2, 14, 3),
                              f'{self.method_prefix}__subsample': [0.5,0.8,1.0],
                              f'{self.method_prefix}__learning_rate': loguniform(1e-4, 1e-1),
                              f'{self.method_prefix}__num_leaves': [2,5,10,15,20,30],
                              f'{self.method_prefix}__boosting_type':['gbdt','dart']
                            },
                    'mlp' :{f'{self.method_prefix}': [MLPRegressorWrapper()],
                              f'{self.method_prefix}__hidden_layers':range(1,5),
                              f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__hidden_layers_size': [10,20,50,100],
                              f'{self.method_prefix}__learning_rate_init': loguniform(1e-5,1e-1),
                              f'{self.method_prefix}__max_iter': [50,100,200,400]
                            },
                    'svr': {f'{self.method_prefix}': [SVR(tol=1e-2,max_iter=500)], 
                            f'{self.method_prefix}__C':  loguniform(1e-4,1e2), 
                            f'{self.method_prefix}__epsilon':  loguniform(1e-4,1e2),
                            f'{self.method_prefix}__gamma': loguniform(1e-3,1e2)
                            #,f'{self.method_prefix}__kernel': ['linear', 'rbf']
                           },
                    'rfr' : { f'{self.method_prefix}': [RandomForestRegressor(n_jobs=rfr_threads)],
                              f'{self.method_prefix}__criterion':['squared_error'],
                             f'{self.method_prefix}__random_state':random_state_list,
                             f'{self.method_prefix}__n_estimators' : n_estimators
                             ,f'{self.method_prefix}__max_depth' : [3,4,5,8]
                             ,f'{self.method_prefix}__min_samples_split' : [2,4,8]
                             ,f'{self.method_prefix}__min_samples_leaf':loguniform(1e-3,2e-1) 
                             #,f'{self.method_prefix}__bootstrap': [True, False]
                            },
                    'ada':{ f'{self.method_prefix}': [AdaBoostRegressor()],
                             f'{self.method_prefix}__n_estimators' : n_estimators,
                             f'{self.method_prefix}__random_state':random_state_list
                             ,f'{self.method_prefix}__learning_rate' : loguniform(1e-3,1e2)
                             #,f'{self.method_prefix}__loss' : ['linear', 'square', 'exponential']
                            },
                    'bayesianridge': {  f'{self.method_prefix}': [BayesianRidge()],
                                   f'{self.method_prefix}__alpha_1':loguniform(1e-8,1e-1),
                                    f'{self.method_prefix}__alpha_2':loguniform(1e-8,1e-1),
                                    f'{self.method_prefix}__lambda_1':loguniform(1e-8,1e-1),
                                    f'{self.method_prefix}__lambda_2':loguniform(1e-8,1e-1)
                                  },
                    'dtr': {  f'{self.method_prefix}': [DecisionTreeRegressor()],
                              f'{self.method_prefix}__criterion':['squared_error'],
                              f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__max_depth':[3,4,5,8],
                              f'{self.method_prefix}__min_samples_split': range(2,10,2),
                              f'{self.method_prefix}__min_samples_leaf':loguniform(1e-3,2e-1)  
                           },
                    'sgdr': {  f'{self.method_prefix}': [SGDRegressor()],
                                   f'{self.method_prefix}__alpha':loguniform(1e-6,1e1),
                                   f'{self.method_prefix}__random_state':random_state_list,
                                   f'{self.method_prefix}__max_iter':[100,500,1000,3000],
                                   f'{self.method_prefix}__tol': loguniform(1e-6,1e-2),
                                   f'{self.method_prefix}__penalty': ['l2','l1','elasticnet']
                                  },
                    'kernelridge': {  f'{self.method_prefix}': [KernelRidge(kernel='rbf')],
                                   f'{self.method_prefix}__alpha':loguniform(1e-4,1e2),
                                   f'{self.method_prefix}__gamma': loguniform(1e-3,1e1)
                                  },

                    'gp': {  f'{self.method_prefix}': [GaussianProcessRegressor()],
                                   f'{self.method_prefix}__kernel':gp_kernels,
                                   f'{self.method_prefix}__random_state':random_state_list
                                  },
                }
        else:
            if hyperopt_defaults:
                self.methods={
                    'lasso':{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_lasso',[Lasso(tol=1e-2,max_iter=500)]),
                             f'{self.method_prefix}__alpha':hp.choice(f'{self.method_prefix}__alpha_lasso',np.logspace(-5, 2, 10)),
                             f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_lasso',random_state_list)},
                    'huber':{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_huber',[HuberRegressor(tol=1e-2,max_iter=500)]),
                             f'{self.method_prefix}__alpha':hp.choice(f'{self.method_prefix}__alpha_huber',np.logspace(-5, 2, 10)),
                             f'{self.method_prefix}__epsilon':hp.choice(f'{self.method_prefix}__epsilon_huber',[1.0,1.15,1.3,1.35,1.5,1.7])},
                    'pls' : {f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_pls',[PLSRegression(tol=1e-3,max_iter=500)]),
                             f'{self.method_prefix}__n_components':hp.choice(f'{self.method_prefix}__n_components_pls',pls_n_components)  },
                    'xgb' :{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_xgb',[xgb.XGBRegressor(objective ='reg:squarederror',eval_metric='mlogloss',verbosity =1,nthread=xgb_threads)]),
                              f'{self.method_prefix}__eval_metric':hp.choice(f'{method_prefix}__eval_metric_xgb', ['rmse','mae','logloss']),
                              f'{self.method_prefix}__n_estimators': hp.choice(f'{self.method_prefix}__n_estimators_xgb',n_estimators),
                              f'{self.method_prefix}__max_depth': hp.choice(f'{self.method_prefix}__max_depth_xgb',range(3,5)),
                             f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_xgb',random_state_list),
                              f'{self.method_prefix}__min_child_weight': hp.choice(f'{self.method_prefix}__min_child_weight_xgb',range(2, 14, 3)),
                              f'{self.method_prefix}__gamma': hp.choice(f'{self.method_prefix}__gamma_xgb',[0.5, 1, 1.5, 2, 5]),
                              f'{self.method_prefix}__learning_rate': hp.choice(f'{self.method_prefix}__learning_rate_xgb',np.logspace(-4, -1, 4)),
                              f'{self.method_prefix}__subsample': hp.choice(f'{self.method_prefix}__subsample_xgb',[0.5,0.8,1.0]),
                              f'{self.method_prefix}__colsample_bytree': hp.choice(f'{self.method_prefix}__colsample_bytree_xgb',[0.6,0.8,1.0])
                            },
                    'lgbm' :{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_lgbm',[LGBMRegressor(n_jobs=xgb_threads,verbosity=-1)]),
                              f'{self.method_prefix}__n_estimators':hp.choice(f'{self.method_prefix}_lgbm__n_estimators', n_estimators),
                              f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}_lgbm__random_state',random_state_list),
                              f'{self.method_prefix}__max_depth': hp.choice(f'{self.method_prefix}_lgbm__max_depth',range(3,8)),
                              f'{self.method_prefix}__reg_alpha': hp.choice(f'{self.method_prefix}_lgbm__reg_alpha',np.logspace(-4, 0, 5)),
                              f'{self.method_prefix}__reg_lambda': hp.choice(f'{self.method_prefix}_lgbm__reg_lambda',np.logspace(-4, 0, 5)),
                              f'{self.method_prefix}__min_child_samples': hp.choice(f'{self.method_prefix}_lgbm__min_child_samples',range(2, 14, 3)),
                              f'{self.method_prefix}__subsample': hp.choice(f'{self.method_prefix}_lgbm__subsample',[0.5,0.8,1.0]),
                              f'{self.method_prefix}__learning_rate': hp.choice(f'{self.method_prefix}_lgbm__learning_rate',np.logspace(-4, -1, 4)),
                              f'{self.method_prefix}__num_leaves': hp.choice(f'{self.method_prefix}_lgbm__num_leaves',[2,5,10,15,20,30]),
                              f'{self.method_prefix}__boosting_type':hp.choice(f'{self.method_prefix}_lgbm__boosting_type',['gbdt','dart'])
                            },
                    'mlp' :{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_mlp',[MLPRegressorWrapper()]),
                              f'{self.method_prefix}__hidden_layers':hp.choice(f'{self.method_prefix}_mlp__hidden_layers', range(1,5)),
                              f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}_mlp__random_state',random_state_list),
                              f'{self.method_prefix}__hidden_layers_size': hp.choice(f'{self.method_prefix}_mlp__hidden_layers_size',[10,20,50,100]),
                              f'{self.method_prefix}__learning_rate_init': hp.choice(f'{self.method_prefix}_mlp__learning_rate_init',[1e-5,1e-4,1e-3,1e-2,1e-1]),
                              f'{self.method_prefix}__max_iter': hp.choice(f'{self.method_prefix}_mlp__max_iter',[50,100,200,400])
                            },
                    'svr': {f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_svr',[SVR(tol=1e-2,max_iter=500)]),
                            f'{self.method_prefix}__C':  hp.choice(f'{self.method_prefix}__C_svr',np.logspace(-3, 3, 6)),
                            f'{self.method_prefix}__epsilon':  hp.choice(f'{self.method_prefix}__epsilon_svr',np.logspace(-3, 3, 6)),
                            f'{self.method_prefix}__gamma': hp.choice(f'{self.method_prefix}__gamma',np.logspace(-2, 2, 5))
                            #,f'{self.method_prefix}__kernel': ['linear', 'rbf']
                           },
                   # 'rfr' : { f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_rfr',[RandomForestRegressor(n_jobs=rfr_threads)]),
                   #           f'{self.method_prefix}__criterion':hp.choice(f'{self.method_prefix}__criterion_rfr',['squared_error','poisson']),
                   #          f'{self.method_prefix}__n_estimators' : hp.choice(f'{self.method_prefix}__n_estimators_rfr',n_estimators),
                   #          f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_rfr',random_state_list)
                   #          #,f'{self.method_prefix}__max_features' : ["auto", "sqrt", "log2"]
                   #          ,f'{self.method_prefix}__max_depth' : hp.choice(f'{self.method_prefix}__max_depth_rfr',[3,4,5,8])
                   #          ,f'{self.method_prefix}__min_samples_split' : hp.choice(f'{self.method_prefix}__min_samples_split_rfr',[2,4,8])
                   #          ,f'{self.method_prefix}__min_samples_leaf':hp.choice(f'{self.method_prefix}__min_samples_leaf_rfr',np.logspace(-4, -1, 8)) 
                   #          #,f'{self.method_prefix}__bootstrap': [True, False]
                   #         },
                   # 'ada':{ f'{self.method_prefix}':hp.choice(f'{self.method_prefix}_ada', [AdaBoostRegressor()]),
                   #          f'{self.method_prefix}__n_estimators' :hp.choice(f'{self.method_prefix}__n_estimators_ada', n_estimators),
                   #          f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_ada',random_state_list)
                   #          ,f'{self.method_prefix}__learning_rate' : hp.choice(f'{self.method_prefix}__learning_rate_ada',np.logspace(-3, 2, 10))
                   #         },
                    'bayesianridge': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}__br',[BayesianRidge()]),
                                   f'{self.method_prefix}__alpha_1':hp.choice(f'{self.method_prefix}__alpha_1_rb',np.logspace(-8, -1, 4)),
                                    f'{self.method_prefix}__alpha_2':hp.choice(f'{self.method_prefix}__alpha_2_br',np.logspace(-8, -1, 4)),
                                    f'{self.method_prefix}__lambda_1':hp.choice(f'{self.method_prefix}__lambda_1_br',np.logspace(-8, -1, 4)),
                                    f'{self.method_prefix}__lambda_2':hp.choice(f'{self.method_prefix}__lambda_2_br',np.logspace(-8, -1, 4))
                                  },
                    'dtr': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}__dtr',[DecisionTreeRegressor()]),
                              f'{self.method_prefix}__criterion':hp.choice(f'{self.method_prefix}__criterion_dtr',['squared_error']),
                              f'{self.method_prefix}__max_depth':hp.choice(f'{self.method_prefix}__max_depth_dtr',[3,4,5,8]),
                              f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_dtr',random_state_list),
                              f'{self.method_prefix}__min_samples_split': hp.choice(f'{self.method_prefix}__min_samples_split_dtr',range(2,10,2)),
                              f'{self.method_prefix}__min_samples_leaf':hp.choice(f'{self.method_prefix}__min_samples_leaf_dtr',np.logspace(-4, -1, 8))  
                           },
                    'sgdr': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_sgdr',[SGDRegressor()]),
                                   f'{self.method_prefix}__alpha':hp.choice(f'{self.method_prefix}__alpha_sgdr',np.logspace(-6, 1, 8)),
                                   f'{self.method_prefix}__max_iter':hp.choice(f'{self.method_prefix}__max_iter_sgdr',[100,500,1000,3000]),
                                   f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_sgdr',random_state_list),
                                   f'{self.method_prefix}__tol': hp.choice(f'{self.method_prefix}__tol_sgdr',np.logspace(-6, -2, 8)),
                                   f'{self.method_prefix}__penalty': hp.choice(f'{self.method_prefix}__penalty_sgdr',['l2','l1','elasticnet'])
                                  },
                    'kernelridge': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_kernelridge',[KernelRidge(kernel='rbf')]),
                                   f'{self.method_prefix}__alpha':hp.choice(f'{self.method_prefix}__alpha_kernelridge',np.logspace(-4, 2, 15)),
                                   f'{self.method_prefix}__gamma': hp.choice(f'{self.method_prefix}__gamma_kernelridge',np.logspace(-4, 2, 5))
                                  },
                    'gp': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_gp',[GaussianProcessRegressor()]),
                                f'{self.method_prefix}__kernel':hp.choice(f'{self.method_prefix}__kernel_gp',gp_kernels),
                                f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_gp',random_state_list)
                            },
                }

            else:
                self.methods={
                    'lasso':{f'{self.method_prefix}': [Lasso(tol=1e-2,max_iter=500)],
                             f'{self.method_prefix}__alpha':np.logspace(-5, 2, 10),
                             f'{self.method_prefix}__random_state':random_state_list},
                    'huber':{f'{self.method_prefix}': [HuberRegressor(tol=1e-2,max_iter=500)],
                             f'{self.method_prefix}__alpha':np.logspace(-5, 2, 10),
                             f'{self.method_prefix}__epsilon':[1.0,1.15,1.3,1.35,1.5,1.7]},
                    'pls' : {f'{self.method_prefix}': [PLSRegression(tol=1e-3,max_iter=500)],
                             f'{self.method_prefix}__n_components':pls_n_components  },
                    'xgb' :{f'{self.method_prefix}': [xgb.XGBRegressor(objective ='reg:squarederror',verbosity =1,nthread=xgb_threads)],
                              f'{self.method_prefix}__eval_metric': ['rmse','mae','logloss'],
                              f'{self.method_prefix}__n_estimators': n_estimators,
                              f'{self.method_prefix}__max_depth': range(3,5),
                             f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__min_child_weight': range(2, 14, 3),
                              f'{self.method_prefix}__gamma': [0.5, 1, 1.5, 2, 5],
                              f'{self.method_prefix}__learning_rate': np.logspace(-4, -1, 4),
                              f'{self.method_prefix}__subsample': [0.5,0.8,1.0],
                              f'{self.method_prefix}__colsample_bytree': [0.6,0.8,1.0]
                            },
                    'lgbm' :{f'{self.method_prefix}': [LGBMRegressor(n_jobs=xgb_threads,verbosity=-1)],
                              f'{self.method_prefix}__n_estimators': n_estimators,
                              f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__max_depth': range(3,8),
                              f'{self.method_prefix}__reg_alpha': np.logspace(-4, 0, 5),
                              f'{self.method_prefix}__reg_lambda': np.logspace(-4, 0, 5),
                              f'{self.method_prefix}__min_child_samples': range(2, 14, 3),
                              f'{self.method_prefix}__subsample': [0.5,0.8,1.0],
                              f'{self.method_prefix}__learning_rate': np.logspace(-4, -1, 4),
                              f'{self.method_prefix}__num_leaves': [2,5,10,15,20,30],
                              f'{self.method_prefix}__boosting_type':['gbdt','dart']
                            },
                    'mlp' :{f'{self.method_prefix}': [MLPRegressorWrapper()],
                              f'{self.method_prefix}__hidden_layers':range(1,5),
                              f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__hidden_layers_size': [10,20,50,100],
                              f'{self.method_prefix}__learning_rate_init': np.logspace(-5, -1, 6),
                              f'{self.method_prefix}__max_iter': [50,100,200,400]
                            },
                    'svr': {f'{self.method_prefix}': [SVR(tol=1e-2,max_iter=500)],
                            f'{self.method_prefix}__C':  np.logspace(-3, 3, 6),
                            f'{self.method_prefix}__epsilon':  np.logspace(-3, 3, 6),
                            f'{self.method_prefix}__gamma': np.logspace(-3, 3,6)
                            #,f'{self.method_prefix}__kernel': ['linear', 'rbf']
                           },
                    'rfr' : { f'{self.method_prefix}': [RandomForestRegressor(n_jobs=rfr_threads)],
                              f'{self.method_prefix}__criterion':['squared_error'],
                             f'{self.method_prefix}__n_estimators' : n_estimators,
                             f'{self.method_prefix}__random_state':random_state_list
                             #,f'{self.method_prefix}__max_features' : ["auto", "sqrt", "log2"]
                             ,f'{self.method_prefix}__max_depth' : [3,4,5,8]
                             ,f'{self.method_prefix}__min_samples_split' : [2,4,8]
                             ,f'{self.method_prefix}__min_samples_leaf':np.logspace(-4, -1, 8) 
                             #,f'{self.method_prefix}__bootstrap': [True, False]
                            },
                    'ada':{ f'{self.method_prefix}': [AdaBoostRegressor()],
                             f'{self.method_prefix}__n_estimators' : n_estimators,
                             f'{self.method_prefix}__random_state':random_state_list
                             ,f'{self.method_prefix}__learning_rate' : np.logspace(-3, 2, 10)
                            },
                    'bayesianridge': {  f'{self.method_prefix}': [BayesianRidge()],
                                   f'{self.method_prefix}__alpha_1':np.logspace(-8, -1, 4),
                                    f'{self.method_prefix}__alpha_2':np.logspace(-8, -1, 4),
                                    f'{self.method_prefix}__lambda_1':np.logspace(-8, -1, 4),
                                    f'{self.method_prefix}__lambda_2':np.logspace(-8, -1, 4)
                                  },
                    'dtr': {  f'{self.method_prefix}': [DecisionTreeRegressor()],
                              f'{self.method_prefix}__criterion':['squared_error'],
                              f'{self.method_prefix}__max_depth':[3,4,5,8],
                              f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__min_samples_split': range(2,10,2),
                              f'{self.method_prefix}__min_samples_leaf':np.logspace(-4, -1, 8)  
                           },
                    'sgdr': {  f'{self.method_prefix}': [SGDRegressor()],
                                   f'{self.method_prefix}__alpha':np.logspace(-6, 2, 8),
                                   f'{self.method_prefix}__max_iter':[100,500,1000,3000],
                                   f'{self.method_prefix}__random_state':random_state_list,
                                   f'{self.method_prefix}__tol': np.logspace(-6, -2, 8),
                                   f'{self.method_prefix}__penalty': ['l2','l1','elasticnet']
                                  },
                    'kernelridge': {  f'{self.method_prefix}': [KernelRidge(kernel='rbf')],
                                   f'{self.method_prefix}__alpha':np.logspace(-4, 2, 15),
                                   f'{self.method_prefix}__gamma': np.logspace(-4, 2, 6)
                                  },
                    'gp': {  f'{self.method_prefix}': [GaussianProcessRegressor()],
                                f'{self.method_prefix}__kernel':gp_kernels,
                                f'{self.method_prefix}__random_state':random_state_list
                            },
                }

###########################
class ClassifierArchive(MethodArchive):
    """
    derived class of MethodArchive with the defaults classifier estimators and their parameters
    """
    def __init__(self,
                 method_prefix='clf',
                 distribution_defaults=False,
                 hyperopt_defaults=False,
                 xgb_threads=4,
                 rfr_threads=-1,
                 method_jobs=None,
                 n_estimators=None,
                 gp_kernels=None,
                 random_state=42):
        """
        Initialization
        
        Args:
             method_prefix: the method prefix
             distribution_defaults: boolean for default distributional parameters
             hyperopt_defaults: boolean for default hyperopt parameters
             xgb_threads: number of threads for XGBoost
             rfr_threads: number of threads for RandomForest
             method_jobs: number of threads for the methods
             n_estimators: number of estimators for tree based ensembles (Random forest, XGboost, Adaboost)
             pls_n_components: number of components for pls
             gp_kernels: gaussian process kernels
             random_state: the random state init value
        """
        
        self.method_prefix=method_prefix
        if n_estimators is None : n_estimators=[100,250,500,1000]
        if gp_kernels is None: gp_kernels=[None, DotProduct() + WhiteKernel(),Matern()]
        if isinstance(random_state, (list)):
            random_state_list=random_state
        else:
            random_state_list=[random_state]
        self.hyperopt_defaults=hyperopt_defaults
        
        if method_jobs is None:
            if xgb_threads is not None:
                method_jobs = xgb_threads
            elif rfr_threads is not None:
                method_jobs=rfr_threads
            else:
                method_jobs=2
                rfr_threads=method_jobs
                xgb_threads=method_jobs
        if rfr_threads is None:
            rfr_threads=method_jobs
        if xgb_threads is None:
            xgb_threads=method_jobs
        
        if distribution_defaults:
            if hyperopt_defaults:
                self.methods={
                'lr':{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_lr',[LogisticRegression(solver='lbfgs', max_iter=500)]),
                            f'{self.method_prefix}__C':hp.loguniform(f'{self.method_prefix}__C_lr',-3, 2),
                            f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_lr',random_state_list)},
                'knn':{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_knn',[KNeighborsClassifier(n_jobs=method_jobs)]),
                       f'{self.method_prefix}__n_neighbors':hp.choice(f'{self.method_prefix}__n_neighbors_knn',[2,3,5,10])},
                'dtc' : {f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_dtc',[DecisionTreeClassifier()]),
                          f'{self.method_prefix}__criterion':hp.choice(f'{self.method_prefix}__criterion_dtc',['gini','entropy']),
                          f'{self.method_prefix}__max_depth':hp.choice(f'{self.method_prefix}__max_depth_dtc',range(3,8)),
                          f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_dtc',random_state_list),
                          f'{self.method_prefix}__min_samples_split':hp.choice(f'{self.method_prefix}__min_samples_split_dtc',range(2,10,2)),
                          f'{self.method_prefix}__min_samples_leaf':hp.loguniform(f'{self.method_prefix}__min_samples_leaf_dtc',-4, -1) },
                'xgb' :{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_xgb',[xgb.XGBClassifier(objective ='binary:logistic',
                                                                    verbosity =1,nthread=xgb_threads, random_state=random_state_list[0] )]),
                          f'{self.method_prefix}__eval_metric':hp.choice(f'{self.method_prefix}__eval_metric_xgb', ['auc','mlogloss']),
                          f'{self.method_prefix}__n_estimators':hp.choice(f'{self.method_prefix}__n_estimators_xgb', n_estimators),
                          f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_xgb',random_state_list),
                          f'{self.method_prefix}__max_depth':hp.choice(f'{self.method_prefix}__max_depth_xgb', range(3,8)),
                          f'{self.method_prefix}__min_child_weight': hp.quniform(f'{self.method_prefix}__min_child_weight_xgb',2, 14, 2),
                          f'{self.method_prefix}__gamma': hp.quniform(f'{self.method_prefix}__gamma_xgb',0.5, 5,0.5),
                          f'{self.method_prefix}__learning_rate': hp.loguniform(f'{self.method_prefix}__learning_rate_xgb',-4, -1),
                          f'{self.method_prefix}__subsample': hp.uniform(f'{self.method_prefix}__subsample_xgb',0.5,1.0),
                          f'{self.method_prefix}__colsample_bytree': hp.uniform(f'{self.method_prefix}__colsample_bytree_xgb',0.6,1.0)
                        },
                'lgbm' :{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_lgbm',[LGBMClassifier(n_jobs=xgb_threads,verbosity=-1)]),
                              f'{self.method_prefix}__n_estimators':hp.choice(f'{self.method_prefix}_lgbm__n_estimators', n_estimators),
                              f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}_lgbm__random_state',random_state_list),
                              f'{self.method_prefix}__max_depth': hp.choice(f'{self.method_prefix}_lgbm__max_depth',range(3,8)),
                              f'{self.method_prefix}__reg_alpha': hp.loguniform(f'{self.method_prefix}_lgbm__reg_alpha',-4, 1),
                              f'{self.method_prefix}__reg_lambda': hp.loguniform(f'{self.method_prefix}_lgbm__reg_lambda',-4, 1),
                              f'{self.method_prefix}__min_child_samples': hp.choice(f'{self.method_prefix}_lgbm__min_child_samples',range(2, 14, 3)),
                              f'{self.method_prefix}__subsample': hp.choice(f'{self.method_prefix}_lgbm__subsample',[0.5,0.8,1.0]),
                              f'{self.method_prefix}__learning_rate': hp.loguniform(f'{self.method_prefix}_lgbm__learning_rate',-4, -1),
                              f'{self.method_prefix}__num_leaves': hp.choice(f'{self.method_prefix}_lgbm__num_leaves',[2,5,10,15,20,30]),
                              f'{self.method_prefix}__boosting_type':hp.choice(f'{self.method_prefix}_lgbm__boosting_type',['gbdt','dart'])
                            },
                    'mlp' :{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_mlp',[MLPClassifierWrapper()]),
                              f'{self.method_prefix}__hidden_layers':hp.choice(f'{self.method_prefix}_mlp__hidden_layers', range(1,5)),
                              f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}_mlp__random_state',random_state_list),
                              f'{self.method_prefix}__hidden_layers_size': hp.choice(f'{self.method_prefix}_mlp__hidden_layers_size',[10,20,50,100]),
                              f'{self.method_prefix}__learning_rate_init': hp.loguniform(f'{self.method_prefix}_mlp__learning_rate_init',-5, -1),
                              f'{self.method_prefix}__max_iter': hp.choice(f'{self.method_prefix}_mlp__max_iter',[50,100,200,400])
                            },

                'SVC': {f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_svc',[SVC(probability=True)]),
                        f'{self.method_prefix}__C':  hp.loguniform(f'{self.method_prefix}__C_svc',-2, 2),
                        f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_svc',random_state_list),
                        f'{self.method_prefix}__gamma': hp.loguniform(f'{self.method_prefix}__gamma_svc',-2, 2)
                        #,f'{self.method_prefix}__kernel': ['linear', 'rbf']
                       },
                #'rfc' : { f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_rfc',[RandomForestClassifier(n_jobs=rfr_threads)]),
                #          f'{self.method_prefix}__criterion':hp.choice(f'{self.method_prefix}__criterion_rfc',['gini','entropy']),
                #         f'{self.method_prefix}__n_estimators' : hp.choice(f'{self.method_prefix}__n_estimators_rfc',n_estimators),
                #            f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_rfc',random_state_list)
                #         ,f'{self.method_prefix}__max_depth' : hp.choice(f'{self.method_prefix}__max_depth_rfc',[3,4,5,8])
                #         #,f'{self.method_prefix}__max_features' : ["auto", "sqrt", "log2"]
                #         ,f'{self.method_prefix}__min_samples_split' : hp.choice(f'{self.method_prefix}__min_samples_split_rfc',[2,4,8])
                #         ,f'{self.method_prefix}__min_samples_leaf':hp.choice(f'{self.method_prefix}__min_samples_leaf_rfc',np.logspace(-4, -1, 8))
                #         #f'{self.method_prefix}__bootstrap': [True, False]
                #        }, 
                #'ada':{ f'{self.method_prefix}': hp.choice(f'{self.method_prefix}__ada',[AdaBoostClassifier()]),
                #         f'{self.method_prefix}__n_estimators' : hp.choice(f'{self.method_prefix}__n_estimators',n_estimators),
                #         f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state',random_state_list)
                #         ,f'{self.method_prefix}__learning_rate' : hp.choice(f'{self.method_prefix}__learning_rate_ada',np.logspace(-3, 2, 5))
                #         #,f'{self.method_prefix}__loss' : ['linear', 'square', 'exponential']
                #        },
                'lda':{ f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_lda',[LinearDiscriminantAnalysis()])
                        },
                'qda':{ f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_qda',[QuadraticDiscriminantAnalysis()])
                        },
                'sgdc': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_sgd',[SGDClassifier(loss='modified_huber')]),
                            f'{self.method_prefix}__alpha':hp.loguniform(f'{self.method_prefix}__alpha_sgd',-6, 1),
                            f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_sgd',random_state_list),
                            f'{self.method_prefix}__max_iter':hp.choice(f'{self.method_prefix}__max_iter_sgd',[100,500,1000,3000]),
                            f'{self.method_prefix}__tol': hp.loguniform(f'{self.method_prefix}__tol_sgd',-6, -2),
                            f'{self.method_prefix}__penalty': hp.choice(f'{self.method_prefix}__penalty_sgd',['l2','l1','elasticnet'])
                            },
                'gpc': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}__gpc',[GaussianProcessClassifier()]),
                          f'{self.method_prefix}__kernel':hp.choice(f'{self.method_prefix}__kernel_gpc',gp_kernels),
                            f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_gpc',random_state_list)
                              },
                'nb': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_nb',[GaussianNB()])
                      },
               }
            else:
                self.methods={
                    'lr':{f'{self.method_prefix}': [LogisticRegression(solver='lbfgs', max_iter=500)],
                          f'{self.method_prefix}__C':loguniform(1e-3,1e2),
                                f'{self.method_prefix}__random_state':random_state_list},
                    'knn':{f'{self.method_prefix}': [KNeighborsClassifier(n_jobs=method_jobs)],
                           f'{self.method_prefix}__n_neighbors':[2,3,5,10]},
                    'dtc' : {f'{self.method_prefix}': [DecisionTreeClassifier()],
                              f'{self.method_prefix}__criterion':['gini','entropy'],
                              f'{self.method_prefix}__max_depth':[3,4,5,8],
                                f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__min_samples_split': range(2,10,2),
                              f'{self.method_prefix}__min_samples_leaf':loguniform(1e-3,2e-1) 
                            },
                    'xgb' :{f'{self.method_prefix}': [xgb.XGBClassifier(objective ='binary:logistic',verbosity =1,nthread=xgb_threads, random_state=random_state_list[0])],
                              f'{self.method_prefix}__eval_metric':['auc','mlogloss'],
                              f'{self.method_prefix}__n_estimators': n_estimators,
                              f'{self.method_prefix}__max_depth': range(3,5),
                              f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__min_child_weight': range(2, 14, 3),
                              f'{self.method_prefix}__gamma': [0.5, 1, 1.5, 2, 5],
                              f'{self.method_prefix}__learning_rate': loguniform(1e-3,2e-1),
                              f'{self.method_prefix}__subsample': [0.5,0.8,1.0],
                              f'{self.method_prefix}__colsample_bytree': [0.6,0.8,1.0]
                            },
                    'lgbm' :{f'{self.method_prefix}': [LGBMClassifier(n_jobs=xgb_threads,verbosity=-1)],
                              f'{self.method_prefix}__n_estimators': n_estimators,
                              f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__max_depth': range(3,8),
                              f'{self.method_prefix}__reg_alpha': loguniform(1e-4, 1e1),
                              f'{self.method_prefix}__reg_lambda': loguniform(1e-4, 1e1),
                              f'{self.method_prefix}__min_child_samples': range(2, 14, 3),
                              f'{self.method_prefix}__subsample': [0.5,0.8,1.0],
                              f'{self.method_prefix}__learning_rate': loguniform(1e-4, 1e-1),
                              f'{self.method_prefix}__num_leaves': [2,5,10,15,20,30],
                              f'{self.method_prefix}__boosting_type':['gbdt','dart']
                            },
                    'mlp' :{f'{self.method_prefix}': [MLPClassifierWrapper()],
                              f'{self.method_prefix}__hidden_layers':range(1,5),
                              f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__hidden_layers_size': [10,20,50,100],
                              f'{self.method_prefix}__learning_rate_init': loguniform(1e-5,1e-1),
                              f'{self.method_prefix}__max_iter': [50,100,200,400]
                            },
                    'SVC': {f'{self.method_prefix}': [SVC(probability=True)],
                            f'{self.method_prefix}__C':  loguniform(1e-3,1e2),
                            f'{self.method_prefix}__random_state':random_state_list,
                            f'{self.method_prefix}__gamma': loguniform(1e-3,1e2)#,
                            #f'{self.method_prefix}__kernel': ['linear', 'rbf']
                           },
                    'rfc' : { f'{self.method_prefix}': [RandomForestClassifier(n_jobs=rfr_threads)],
                              f'{self.method_prefix}__criterion':['gini','entropy'],
                             f'{self.method_prefix}__n_estimators' : n_estimators,
                             f'{self.method_prefix}__random_state':random_state_list
                             ,f'{self.method_prefix}__max_depth' : [3,4,5,8]
                             ,f'{self.method_prefix}__min_samples_split' : [2,4,8]
                             #,f'{self.method_prefix}__max_features' : ["auto", "sqrt", "log2"]
                             ,f'{self.method_prefix}__min_samples_leaf':loguniform(1e-3,2e-1)
                             #,f'{self.method_prefix}__bootstrap': [True, False]
                            }, 
                    'ada':{ f'{self.method_prefix}': [AdaBoostClassifier()],
                             f'{self.method_prefix}__n_estimators' : n_estimators,
                                f'{self.method_prefix}__random_state':random_state_list
                             ,f'{self.method_prefix}__learning_rate' : loguniform(1e-3,1e2)
                             #,f'{self.method_prefix}__loss' : ['linear', 'square', 'exponential']
                            },
                    'lda':{ f'{self.method_prefix}': [LinearDiscriminantAnalysis()]
                            },
                    'qda':{ f'{self.method_prefix}': [QuadraticDiscriminantAnalysis()]
                            },
                    'sgdc': {  f'{self.method_prefix}': [SGDClassifier(loss='modified_huber')],
                                f'{self.method_prefix}__alpha':loguniform(1e-6,1e1),
                                f'{self.method_prefix}__random_state':random_state_list,
                                f'{self.method_prefix}__max_iter':[100,500,1000,3000],
                                f'{self.method_prefix}__tol': loguniform(1e-6,1e-2),
                                f'{self.method_prefix}__penalty': ['l2','l1','elasticnet']
                                },
                    'gpc': {  f'{self.method_prefix}': [GaussianProcessClassifier()],
                            f'{self.method_prefix}__kernel':gp_kernels,
                            f'{self.method_prefix}__random_state':random_state_list
                            },
                    'nb': {  f'{self.method_prefix}': [GaussianNB()]
                          },
                   }
        else:
            if hyperopt_defaults:
                self.methods={
                'lr':{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_lr',[LogisticRegression(solver='lbfgs', max_iter=500)]),
                            f'{self.method_prefix}__C':hp.choice(f'{self.method_prefix}__C_lr',np.logspace(-3, 2, 6)),
                            f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_lr',random_state_list)},
                'knn':{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_knn',[KNeighborsClassifier(n_jobs=method_jobs)]),
                       f'{self.method_prefix}__n_neighbors':hp.choice(f'{self.method_prefix}__n_neighbors_knn',[2,3,5,10])},
                'dtc' : {f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_dtc',[DecisionTreeClassifier()]),
                          f'{self.method_prefix}__criterion':hp.choice(f'{self.method_prefix}__criterion_dtc',['gini','entropy']),
                          f'{self.method_prefix}__max_depth':hp.choice(f'{self.method_prefix}__max_depth_dtc',[3,4,5,8]),
                          f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_dtc',random_state_list),
                          f'{self.method_prefix}__min_samples_split':hp.choice(f'{self.method_prefix}__min_samples_split_dtc', range(2,10,2)),
                          f'{self.method_prefix}__min_samples_leaf':hp.choice(f'{self.method_prefix}__min_samples_leaf_dtc',np.logspace(-4, -1, 8)) },
                'xgb' :{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_xgb',[xgb.XGBClassifier(objective ='binary:logistic',
                                                                    verbosity =1,nthread=xgb_threads, random_state=random_state_list[0] )]),
                          f'{self.method_prefix}__eval_metric':hp.choice(f'{self.method_prefix}__eval_metric_xgb', ['auc','mlogloss']),
                          f'{self.method_prefix}__n_estimators':hp.choice(f'{self.method_prefix}__n_estimators_xgb', n_estimators),
                          f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_xgb',random_state_list),
                          f'{self.method_prefix}__max_depth':hp.choice(f'{self.method_prefix}__max_depth_xgb', range(3,5)),
                          f'{self.method_prefix}__min_child_weight': hp.choice(f'{self.method_prefix}__min_child_weight_xgb',range(2, 14, 3)),
                          f'{self.method_prefix}__gamma': hp.choice(f'{self.method_prefix}__gamma_xgb',[0.5, 1, 1.5, 2, 5]),
                          f'{self.method_prefix}__learning_rate': hp.choice(f'{self.method_prefix}__learning_rate_xgb',np.logspace(-4, -1, 4)),
                          f'{self.method_prefix}__subsample': hp.choice(f'{self.method_prefix}__subsample_xgb',[0.5,0.8,1.0]),
                          f'{self.method_prefix}__colsample_bytree': hp.choice(f'{self.method_prefix}__colsample_bytree_xgb',[0.6,0.8,1.0])
                        },
                'lgbm' :{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_lgbm',[LGBMClassifier(n_jobs=xgb_threads,verbosity=-1)]),
                            f'{self.method_prefix}__n_estimators':hp.choice(f'{self.method_prefix}_lgbm__n_estimators', n_estimators),
                            f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}_lgbm__random_state',random_state_list),
                            f'{self.method_prefix}__max_depth': hp.choice(f'{self.method_prefix}_lgbm__max_depth',range(3,8)),
                            f'{self.method_prefix}__reg_alpha': hp.choice(f'{self.method_prefix}_lgbm__reg_alpha',np.logspace(-4, 0, 5)),
                            f'{self.method_prefix}__reg_lambda': hp.choice(f'{self.method_prefix}_lgbm__reg_lambda',np.logspace(-4, 0, 5)),
                            f'{self.method_prefix}__min_child_samples': hp.choice(f'{self.method_prefix}_lgbm__min_child_samples',range(2, 14, 3)),
                            f'{self.method_prefix}__subsample': hp.choice(f'{self.method_prefix}_lgbm__subsample',[0.5,0.8,1.0]),
                            f'{self.method_prefix}__learning_rate': hp.choice(f'{self.method_prefix}_lgbm__learning_rate',np.logspace(-4, -1, 4)),
                            f'{self.method_prefix}__num_leaves': hp.choice(f'{self.method_prefix}_lgbm__num_leaves',[2,5,10,15,20,30]),
                            f'{self.method_prefix}__boosting_type':hp.choice(f'{self.method_prefix}_lgbm__boosting_type',['gbdt','dart'])
                        },
                    'mlp' :{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_mlp',[MLPClassifierWrapper()]),
                              f'{self.method_prefix}__hidden_layers':hp.choice(f'{self.method_prefix}_mlp__hidden_layers', range(1,5)),
                              f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}_mlp__random_state',random_state_list),
                              f'{self.method_prefix}__hidden_layers_size': hp.choice(f'{self.method_prefix}_mlp__hidden_layers_size',[10,20,50,100]),
                              f'{self.method_prefix}__learning_rate_init': hp.choice(f'{self.method_prefix}_mlp__learning_rate_init',np.logspace(-5, -1, 6)),
                              f'{self.method_prefix}__max_iter': hp.choice(f'{self.method_prefix}_mlp__max_iter',[50,100,200,400])
                            },
                'SVC': {f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_svc',[SVC(probability=True)]),
                        f'{self.method_prefix}__C':  hp.choice(f'{self.method_prefix}__C_svc',np.logspace(-1, 1, 3)),
                        f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_svc',random_state_list),
                        f'{self.method_prefix}__gamma': hp.choice(f'{self.method_prefix}__gamma_svc',np.logspace(-2, 2, 5))
                        #,f'{self.method_prefix}__kernel': ['linear', 'rbf']
                       },
                #'rfc' : { f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_rfc',[RandomForestClassifier(n_jobs=rfr_threads)]),
                #          f'{self.method_prefix}__criterion':hp.choice(f'{self.method_prefix}__criterion_rfc',['gini','entropy']),
                #         f'{self.method_prefix}__n_estimators' : hp.choice(f'{self.method_prefix}__n_estimators_rfc',n_estimators),
                #            f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_rfc',random_state_list)
                #         ,f'{self.method_prefix}__max_depth' : hp.choice(f'{self.method_prefix}__max_depth_rfc',[3,4,5,8])
                #         #,f'{self.method_prefix}__max_features' : ["auto", "sqrt", "log2"]
                #         ,f'{self.method_prefix}__min_samples_split' : hp.choice(f'{self.method_prefix}__min_samples_split_rfc',[2,4,8])
                #         ,f'{self.method_prefix}__min_samples_leaf':hp.choice(f'{self.method_prefix}__min_samples_leaf_rfc',np.logspace(-4, -1, 8))
                #         #f'{self.method_prefix}__bootstrap': [True, False]
                #        }, 
                #'ada':{ f'{self.method_prefix}': hp.choice(f'{self.method_prefix}__ada',[AdaBoostClassifier()]),
                #         f'{self.method_prefix}__n_estimators' : hp.choice(f'{self.method_prefix}__n_estimators',n_estimators),
                #         f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state',random_state_list)
                #         ,f'{self.method_prefix}__learning_rate' : hp.choice(f'{self.method_prefix}__learning_rate_ada',np.logspace(-3, 2, 5))
                #         #,f'{self.method_prefix}__loss' : ['linear', 'square', 'exponential']
                #        },
                'lda':{ f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_lda',[LinearDiscriminantAnalysis()])
                        },
                'qda':{ f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_qda',[QuadraticDiscriminantAnalysis()])
                        },
                'sgdc': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_sgd',[SGDClassifier(loss='modified_huber')]),
                            f'{self.method_prefix}__alpha':hp.choice(f'{self.method_prefix}__alpha_sgd',np.logspace(-6, 1, 8)),
                            f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_sgd',random_state_list),
                            f'{self.method_prefix}__max_iter':hp.choice(f'{self.method_prefix}__max_iter_sgd',[100,500,1000,3000]),
                            f'{self.method_prefix}__tol': hp.choice(f'{self.method_prefix}__tol_sgd',np.logspace(-6, -2, 4)),
                            f'{self.method_prefix}__penalty': hp.choice(f'{self.method_prefix}__penalty_sgd',['l2','l1','elasticnet'])
                            },
                'gpc': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}__gpc',[GaussianProcessClassifier()]),
                          f'{self.method_prefix}__kernel':hp.choice(f'{self.method_prefix}__kernel_gpc',gp_kernels),
                            f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_gpc',random_state_list)
                              },
                'nb': {  f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_nb',[GaussianNB()])
                      },
               }
            else:
                self.methods={
                    'lr':{f'{self.method_prefix}': [LogisticRegression(solver='lbfgs', max_iter=500)], f'{self.method_prefix}__C':np.logspace(-3, 2, 6),
                                f'{self.method_prefix}__random_state':random_state_list},
                    'knn':{f'{self.method_prefix}': [KNeighborsClassifier(n_jobs=method_jobs)], f'{self.method_prefix}__n_neighbors':[2,3,5,10]},
                    'dtc' : {f'{self.method_prefix}': [DecisionTreeClassifier()],
                              f'{self.method_prefix}__criterion':['gini','entropy'],
                              f'{self.method_prefix}__max_depth':[3,4,5,8],
                                f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__min_samples_split': range(2,10,2),
                              f'{self.method_prefix}__min_samples_leaf':np.logspace(-4, -1, 8) },
                    'xgb' :{f'{self.method_prefix}': [xgb.XGBClassifier(objective ='binary:logistic',
                                                                        verbosity =1,nthread=xgb_threads)],
                              f'{self.method_prefix}__eval_metric':['auc','mlogloss'],
                              f'{self.method_prefix}__n_estimators': n_estimators,
                              f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__max_depth': range(3,5),
                              f'{self.method_prefix}__min_child_weight': range(2, 14, 3),
                              f'{self.method_prefix}__gamma': [0.5, 1, 1.5, 2, 5],
                              f'{self.method_prefix}__learning_rate': np.logspace(-4, -1, 4),
                              f'{self.method_prefix}__subsample': [0.5,0.8,1.0],
                              f'{self.method_prefix}__colsample_bytree': [0.6,0.8,1.0]
                            },
                    'lgbm' :{f'{self.method_prefix}': [LGBMClassifier(n_jobs=xgb_threads,verbosity=-1)],
                              f'{self.method_prefix}__n_estimators': n_estimators,
                              f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__max_depth': range(3,8),
                              f'{self.method_prefix}__reg_alpha': np.logspace(-4, 0, 5),
                              f'{self.method_prefix}__reg_lambda': np.logspace(-4, 0, 5),
                              f'{self.method_prefix}__min_child_samples': range(2, 14, 3),
                              f'{self.method_prefix}__subsample': [0.5,0.8,1.0],
                              f'{self.method_prefix}__learning_rate': np.logspace(-4, -1, 4),
                              f'{self.method_prefix}__num_leaves': [2,5,10,15,20,30],
                              f'{self.method_prefix}__boosting_type':['gbdt','dart']
                            },
                    'mlp' :{f'{self.method_prefix}': [MLPClassifierWrapper()],
                              f'{self.method_prefix}__hidden_layers':range(1,5),
                              f'{self.method_prefix}__random_state':random_state_list,
                              f'{self.method_prefix}__hidden_layers_size': [10,20,50,100],
                              f'{self.method_prefix}__learning_rate_init': np.logspace(-5, -1, 6),
                              f'{self.method_prefix}__max_iter': [50,100,200,400]
                            },
                    'SVC': {f'{self.method_prefix}': [SVC(probability=True)],
                            f'{self.method_prefix}__C':  np.logspace(-1, 1, 3),
                            f'{self.method_prefix}__random_state':random_state_list,
                            f'{self.method_prefix}__gamma': np.logspace(-2, 2, 5)
                            #,f'{self.method_prefix}__kernel': ['linear', 'rbf']
                           },
                    'rfc' : { f'{self.method_prefix}': [RandomForestClassifier(n_jobs=rfr_threads)],
                              f'{self.method_prefix}__criterion':['gini','entropy'],
                             f'{self.method_prefix}__n_estimators' : n_estimators,
                                f'{self.method_prefix}__random_state':random_state_list
                             ,f'{self.method_prefix}__max_depth' : [3,4,5,8]
                             #,f'{self.method_prefix}__max_features' : ["auto", "sqrt", "log2"]
                             ,f'{self.method_prefix}__min_samples_split' : [2,4,8]
                             ,f'{self.method_prefix}__min_samples_leaf':np.logspace(-4, -1, 8)
                             #f'{self.method_prefix}__bootstrap': [True, False]
                            }, 
                    'ada':{ f'{self.method_prefix}': [AdaBoostClassifier()],
                             f'{self.method_prefix}__n_estimators' : n_estimators,
                             f'{self.method_prefix}__random_state':random_state_list
                             ,f'{self.method_prefix}__learning_rate' : np.logspace(-3, 2, 5)
                             #,f'{self.method_prefix}__loss' : ['linear', 'square', 'exponential']
                            },
                    'lda':{ f'{self.method_prefix}': [LinearDiscriminantAnalysis()]
                            },
                    'qda':{ f'{self.method_prefix}': [QuadraticDiscriminantAnalysis()]
                            },
                    'sgdc': {  f'{self.method_prefix}': [SGDClassifier(loss='modified_huber')],
                                f'{self.method_prefix}__alpha':np.logspace(-6, 1, 8),
                                f'{self.method_prefix}__random_state':random_state_list,
                                f'{self.method_prefix}__max_iter':[100,500,1000,3000],
                                f'{self.method_prefix}__tol': np.logspace(-6, -2, 4),
                                f'{self.method_prefix}__penalty': ['l2','l1','elasticnet']
                                },
                    'gpc': {  f'{self.method_prefix}': [GaussianProcessClassifier()],
                              f'{self.method_prefix}__kernel':gp_kernels,
                                f'{self.method_prefix}__random_state':random_state_list
                                  },
                    'nb': {  f'{self.method_prefix}': [GaussianNB()]
                          },
                   }
###########################            
class ReducedimArchive(MethodArchive):
    """
    derived class of MethodArchive with the default dimensionality reduction methods and their parameters
    """
    def __init__(self,
                 method_prefix='reduce_dim',
                 distribution_defaults=False,
                 hyperopt_defaults=False,
                 dim_red_n_components=None,
                 SelectPer=None,
                 variance_thresholds=None,
                random_state=42,
                clf=False):
        """
        Initialization
        
        Args:
             method_prefix: the method prefix
             distribution_defaults: boolean for default distributional parameters
             distribution_defaults: boolean for default hyperopt parameters
             dim_red_n_components: number of components
             SelectPer: percentile selection values
             variance_thresholds: variance thresholds
             random_state: the random state init value
             clf: indicate use of simple classifier for feature selection
        """
        
        self.method_prefix=method_prefix
        if not dim_red_n_components: dim_red_n_components=[10,25,50,100,200,300,500,1000]
        if not SelectPer: SelectPer=[10,25,50,75]
        if not variance_thresholds: variance_thresholds=[0.01,0.05,0.1,0.2]
        if isinstance(random_state, (list)):
            random_state_list=random_state
        else:
            random_state_list=[random_state]
        self.hyperopt_defaults=hyperopt_defaults
        
        if clf:
            scorers= [chi2, f_classif, mutual_info_classif]
            rfe_model=[SVC(kernel="linear", C=1)]
            select_models=[LinearSVC(C=0.01, penalty="l1", dual=False), ExtraTreesClassifier(n_estimators=100, random_state=0)]
        else:
            scorers=[f_regression, mutual_info_regression]
            rfe_model=[SVR(kernel="linear", C=1)]
            select_models=[LinearSVR(C=0.01, loss="epsilon_insensitive", dual=False), ExtraTreesRegressor(n_estimators=100, random_state=0)]
        if distribution_defaults:
            if hyperopt_defaults:
                self.methods={
                'passthrough':      { f'{self.method_prefix}':hp.choice(f'{self.method_prefix}_passthrough',['passthrough'])},
                'pca':              { f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_pca',[PCA()]),
                                      f'{self.method_prefix}__n_components': hp.choice(f'{self.method_prefix}__n_components_pca',dim_red_n_components),
                                      f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_pca',random_state_list)},
                'kpca':             { f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_kpca',[KernelPCA(kernel='rbf')]),
                                      f'{self.method_prefix}__n_components': hp.choice(f'{self.method_prefix}__n_components_kpca',dim_red_n_components),
                                      f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_kpca',random_state_list)},
                'pca+kpca':         { f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_pcakpca',[PCA(),KernelPCA(kernel='rbf')]),
                                      f'{self.method_prefix}__n_components': hp.choice(f'{self.method_prefix}__n_components_pcakpca',dim_red_n_components),
                                      f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_pcakpca',random_state_list)},
                'v_threshold':      { f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_vthres',[VarianceThreshold()]),
                                      f'{self.method_prefix}__threshold'   : hp.uniform(f'{self.method_prefix}__threshold__vthres',1e-2,2e-1)},
                'SelectPercentile': { f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_selperc',[SelectPercentile()]),
                                      f'{self.method_prefix}__percentile'  : hp.quniform(f'{self.method_prefix}__percentile_selperc',1,90,10),
                                       f'{self.method_prefix}__score_func'  : hp.choice(f'{self.method_prefix}__score_func_selperc',scorers)},
                'Kbest':            {f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_Kbest',[SelectKBest()]),
                                     f'{self.method_prefix}__score_func'  : hp.choice(f'{self.method_prefix}__score_func_Kbest',scorers),
                                     f'{self.method_prefix}__k'  : hp.choice(f'{self.method_prefix}__k_Kbest',dim_red_n_components[:5]),
                                    },
                'rfe':            {f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_Kbest',[RFE(estimator=rfe_model[0])]),
                                     f'{self.method_prefix}__estimator'  : hp.choice(f'{self.method_prefix}__estimator_rfe',rfe_model),
                                     f'{self.method_prefix}__n_features_to_select'  : hp.uniform(f'{self.method_prefix}__n_features_to_select_rfe',0.1,0.9),
                                    },
                'frommodel':{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_Kbest',[SelectFromModel(estimator=select_models[0])]),
                                     f'{self.method_prefix}__estimator'  : hp.choice(f'{self.method_prefix}__estimator_frommodel',[select_models[0]]),
                                     f'{self.method_prefix}__max_features'  : hp.uniform(f'{self.method_prefix}__max_features_frommodel',dim_red_n_components[:5]),
                                    }
                }
                if clf:
                    self.methods['lda']={ f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_lda',[LinearDiscriminantAnalysis()])
                            }
            else:
                self.methods={
                    'passthrough':      {f'{self.method_prefix}':['passthrough']},
                    'pca':              { f'{self.method_prefix}': [PCA()],
                                         f'{self.method_prefix}__n_components': dim_red_n_components,
                                        f'{self.method_prefix}__random_state':random_state_list},
                    'kpca':             { f'{self.method_prefix}': [KernelPCA(kernel='rbf')],
                                         f'{self.method_prefix}__n_components': dim_red_n_components,
                                        f'{self.method_prefix}__random_state':random_state_list},
                    'pca+kpca':         { f'{self.method_prefix}': [PCA(),KernelPCA(kernel='rbf')], 
                                         f'{self.method_prefix}__n_components': dim_red_n_components,
                                         f'{self.method_prefix}__random_state':random_state_list},
                    'v_threshold':      { f'{self.method_prefix}': [VarianceThreshold()],
                                         f'{self.method_prefix}__threshold'   : stats.uniform(loc=1e-2,scale=2e-1)},
                    'SelectPercentile': { f'{self.method_prefix}':[SelectPercentile()],
                                          f'{self.method_prefix}__percentile'  : stats.uniform(loc=5,scale=100) ,
                                           f'{self.method_prefix}__score_func'  : scorers},
                    'Kbest':            {f'{self.method_prefix}': [SelectKBest()],
                                         f'{self.method_prefix}__score_func'  : scorers,
                                         f'{self.method_prefix}__k'  : dim_red_n_components[:5],
                                        },
                    'rfe':            {f'{self.method_prefix}': [RFE(estimator=rfe_model[0])],
                                         f'{self.method_prefix}__estimator'  : rfe_model,
                                         f'{self.method_prefix}__n_features_to_select'  : stats.uniform(loc=0.05,scale=1),
                                        },
                    'frommodel':{f'{self.method_prefix}': [SelectFromModel(estimator=select_models[0])],
                                         f'{self.method_prefix}__estimator'  : select_models,
                                         f'{self.method_prefix}__max_features'  : dim_red_n_components[:5],
                                    }
                }
                if clf:
                    self.methods['lda']={ f'{self.method_prefix}': [LinearDiscriminantAnalysis()]
                            }
        else:
            if hyperopt_defaults:
                self.methods={
                'passthrough':      { f'{self.method_prefix}':hp.choice(f'{self.method_prefix}_passthrough',['passthrough'])},
                'pca':              { f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_pca',[PCA()]),
                                      f'{self.method_prefix}__n_components': hp.choice(f'{self.method_prefix}__n_components_pca',dim_red_n_components),
                                      f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_pca',random_state_list)},
                'kpca':             { f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_kpca',[KernelPCA(kernel='rbf')]),
                                      f'{self.method_prefix}__n_components': hp.choice(f'{self.method_prefix}__n_components_kpca',dim_red_n_components),
                                      f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_kpca',random_state_list)},
                'pca+kpca':         { f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_pcakpca',[PCA(),KernelPCA(kernel='rbf')]),
                                      f'{self.method_prefix}__n_components': hp.choice(f'{self.method_prefix}__n_components_pcakpca',dim_red_n_components),
                                      f'{self.method_prefix}__random_state':hp.choice(f'{self.method_prefix}__random_state_pcakpca',random_state_list)},
                'v_threshold':      { f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_vthres',[VarianceThreshold()]),
                                      f'{self.method_prefix}__threshold'   : hp.choice(f'{self.method_prefix}__threshold__vthres',variance_thresholds)},
                'SelectPercentile': { f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_selperc',[SelectPercentile()]),
                                      f'{self.method_prefix}__percentile'  : hp.choice(f'{self.method_prefix}__percentile_selperc',SelectPer),
                                       f'{self.method_prefix}__score_func'  : hp.choice(f'{self.method_prefix}__score_func_selperc',scorers)},
                'Kbest':            {f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_Kbest',[SelectKBest()]),
                                     f'{self.method_prefix}__score_func'  : hp.choice(f'{self.method_prefix}__score_func_Kbest',scorers),
                                     f'{self.method_prefix}__k'  : hp.choice(f'{self.method_prefix}__k_Kbest',dim_red_n_components[:5]),
                                    },
                'rfe':            {f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_Kbest',[RFE(estimator=rfe_model[0])]),
                                     f'{self.method_prefix}__estimator'  : hp.choice(f'{self.method_prefix}__estimator_rfe',rfe_model),
                                     f'{self.method_prefix}__n_features_to_select'  : hp.choice(f'{self.method_prefix}__n_features_to_select_rfe',[0.05,0.1,0.2,0.5,0.7,0.9]),
                                    },
                'frommodel':{f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_Kbest',[SelectFromModel(estimator=select_models[0])]),
                                     f'{self.method_prefix}__estimator'  : hp.choice(f'{self.method_prefix}__estimator_frommodel',[select_models[0]]),
                                     f'{self.method_prefix}__max_features'  : hp.uniform(f'{self.method_prefix}__max_features_frommodel',dim_red_n_components[:5]),
                                    }
                }
                if clf:
                    self.methods['lda']={ f'{self.method_prefix}': hp.choice(f'{self.method_prefix}_lda',[LinearDiscriminantAnalysis()])
                            }
            else:
                self.methods={
                'passthrough':      { f'{self.method_prefix}':['passthrough']},
                'pca':              { f'{self.method_prefix}': [PCA()],
                                      f'{self.method_prefix}__n_components': dim_red_n_components,
                                      f'{self.method_prefix}__random_state':random_state_list},
                'kpca':             { f'{self.method_prefix}': [KernelPCA(kernel='rbf')],
                                      f'{self.method_prefix}__n_components': dim_red_n_components,
                                      f'{self.method_prefix}__random_state':random_state_list},
                'pca+kpca':         { f'{self.method_prefix}': [PCA(),KernelPCA(kernel='rbf')],
                                      f'{self.method_prefix}__n_components': dim_red_n_components,
                                      f'{self.method_prefix}__random_state':random_state_list},
                'v_threshold':      { f'{self.method_prefix}': [VarianceThreshold()],
                                      f'{self.method_prefix}__threshold'   : variance_thresholds},
                'SelectPercentile': { f'{self.method_prefix}':[SelectPercentile()],
                                      f'{self.method_prefix}__percentile'  : SelectPer ,
                                       f'{self.method_prefix}__score_func'  : scorers},
                'Kbest':            {f'{self.method_prefix}': [SelectKBest()],
                                     f'{self.method_prefix}__score_func'  : scorers,
                                     f'{self.method_prefix}__k'  : dim_red_n_components[:5],
                                    },
                'rfe':            {f'{self.method_prefix}': [RFE(estimator=rfe_model[0])],
                                     f'{self.method_prefix}__estimator'  : rfe_model,
                                     f'{self.method_prefix}__n_features_to_select'  : [0.05,0.1,0.2,0.5,0.7,0.9],
                                    },
                'frommodel':     {f'{self.method_prefix}': [SelectFromModel(estimator=select_models[0])],
                                     f'{self.method_prefix}__estimator'  : select_models,
                                     f'{self.method_prefix}__max_features'  : dim_red_n_components[:5]}
                }
                if clf:
                    self.methods['lda']={ f'{self.method_prefix}': [LinearDiscriminantAnalysis()]}
                            
    def get_methods(self,names):
        """
        checks if pca and kpca are in the given names and replaces these by pca+kpca before returning methods        
        
        Args:
             names: list of keys
        
        Returns: 
            list of method dictionaries
        """
        if 'pca' in names and 'kpca' in names:
            names.remove('pca')
            names.remove('kpca')
            names.append('pca+kpca')
        return [self.get_method(name) for name in names]
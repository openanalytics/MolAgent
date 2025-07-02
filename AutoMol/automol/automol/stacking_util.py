"""implementation utilities for the AutoMoL stacking

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""
 
import numpy as np, pandas as pd
from  matplotlib import pyplot as plt


import xgboost as xgb
from lightgbm import LGBMClassifier,LGBMRegressor

 
from .stacking_methodarchive import ClassifierArchive, ReducedimArchive, RegressorArchive
from .stacking import MethodConfigurations
from .param_search import GridSearch, RandomizedSearch, HyperoptSearch
from .stat_util import plot_clf_matrix, plot_clf_auc, plot_clf_confusion, plot_confusion_bars_from_continuos, plot_reg_model_with_error, plot_acc_pre_for_reg, plot_confusion_bars_from_categories, plot_clf_f1, plot_clf_auc

from .clustering import MurckoScaffoldClustering, ButinaSplitReassigned, HierarchicalButina, KmeansForSmiles
from .feature_generators import ECFPGenerator, FeatureGenerator

from hyperopt import hp

def add_xgb_xtimes_hyperopt(method_archive,method_prefix,x,distribution_defaults=False,random_state=None,xgb_threads=4,n_estimators=None,regressor=False):
    """
    duplicates xgb x times with different random_states initializations and returns the corresponding dictionary keys
    
    Args:
         method_archive: the archive class holding the method dictionary
         method_prefix: the method prefix string
         x: number of copies
         distribution_defaults: set to true if you want to use distribution defaults
         random_state: random state initialisation, if value and not a list, a list is generated from this value
         xgb_threads: number of threads of xgboost
         n_estimator: param list of number of estimators
         regressor: boolean to be set to add regressor
    """
    if n_estimators is None : n_estimators=[100,250,500,1000]           
    if random_state is not None and not isinstance(random_state, (list)):
        random_state=[(random_state+(7+random_state)**i)%31393 for i in range(x)]
    if random_state is None:
        random_state=[None for i in range(x)]
    assert len(random_state)==x, 'len of random states is not equal to number of method duplicates'
    clf_list=[]
    if regressor:
        if distribution_defaults:
            for copy_i in range(x):
                m_name=f'xgb_{copy_i}'
                method_archive.methods[m_name]={f'{method_prefix}': hp.choice(f'{method_prefix}_xgb_{copy_i}',[xgb.XGBRegressor(objective ='reg:squarederror',verbosity =1, nthread=xgb_threads, random_state=random_state[copy_i])]),
                              f'{method_prefix}__eval_metric':hp.choice(f'{method_prefix}__eval_metric_xgb_{copy_i}', ['rmse','mae','logloss']),
                              f'{method_prefix}__n_estimators': hp.choice(f'{method_prefix}__n_estimators_xgb_{copy_i}',n_estimators),
                              f'{method_prefix}__max_depth': hp.choice(f'{method_prefix}__max_depth_xgb_{copy_i}',range(3,8)),
                              f'{method_prefix}__min_child_weight': hp.quniform(f'{method_prefix}__min_child_weight_xgb_{copy_i}',2,14,2),
                              f'{method_prefix}__gamma': hp.quniform(f'{method_prefix}__gamma_xgb_{copy_i}',0.5, 5,0.5),
                              f'{method_prefix}__learning_rate': hp.loguniform(f'{method_prefix}__learning_rate_xgb_{copy_i}',-4, -1),
                              f'{method_prefix}__subsample': hp.uniform(f'{method_prefix}__subsample_xgb_{copy_i}',0.5,1.0),
                              f'{method_prefix}__colsample_bytree': hp.uniform(f'{method_prefix}__colsample_bytree_xgb_{copy_i}',0.6,1.0)
                            }
                clf_list.append(m_name)
        else:
            for copy_i in range(x):
                m_name=f'xgb_{copy_i}'
                method_archive.methods[m_name]={f'{method_prefix}': hp.choice(f'{method_prefix}_xgb_{copy_i}',[xgb.XGBRegressor(objective ='reg:squarederror',verbosity =1, nthread=xgb_threads, random_state=random_state[copy_i])]),
                              f'{method_prefix}__eval_metric':hp.choice(f'{method_prefix}__eval_metric_xgb_{copy_i}', ['rmse','mae','logloss']),
                              f'{method_prefix}__n_estimators': hp.choice(f'{method_prefix}__n_estimators_xgb_{copy_i}',n_estimators),
                              f'{method_prefix}__max_depth': hp.choice(f'{method_prefix}__max_depth_xgb_{copy_i}',range(3,5)),
                              f'{method_prefix}__min_child_weight': hp.choice(f'{method_prefix}__min_child_weight_xgb_{copy_i}',range(2, 14, 3)),
                              f'{method_prefix}__gamma': hp.choice(f'{method_prefix}__gamma_xgb_{copy_i}',[0.5, 1, 1.5, 2, 5]),
                              f'{method_prefix}__learning_rate': hp.choice(f'{method_prefix}__learning_rate_xgb_{copy_i}',np.logspace(-4, -1, 4)),
                              f'{method_prefix}__subsample': hp.choice(f'{method_prefix}__subsample_xgb_{copy_i}',[0.5,0.8,1.0]),
                              f'{method_prefix}__colsample_bytree': hp.choice(f'{method_prefix}__colsample_bytree_xgb_{copy_i}',[0.6,0.8,1.0])
                            }
                clf_list.append(m_name)
    else:
        if distribution_defaults:
            for copy_i in range(x):
                m_name=f'xgb_{copy_i}'
                method_archive.methods[m_name]={f'{method_prefix}': hp.choice(f'{method_prefix}_xgb_{copy_i}',[xgb.XGBClassifier(objective ='binary:logistic',
                                                                        verbosity =1,nthread=xgb_threads, random_state=random_state[copy_i] )]),
                              f'{method_prefix}__eval_metric':hp.choice(f'{method_prefix}__eval_metric_xgb_{copy_i}', ['auc','mlogloss']),
                              f'{method_prefix}__n_estimators':hp.choice(f'{method_prefix}__n_estimators_xgb_{copy_i}', n_estimators),
                              f'{method_prefix}__max_depth':hp.choice(f'{method_prefix}__max_depth_xgb_{copy_i}', range(3,8)),
                              f'{method_prefix}__min_child_weight': hp.quniform(f'{method_prefix}__min_child_weight_xgb_{copy_i}',2, 14, 2),
                              f'{method_prefix}__gamma': hp.quniform(f'{method_prefix}__gamma_xgb_{copy_i}',0.5, 5,0.5),
                              f'{method_prefix}__learning_rate': hp.loguniform(f'{method_prefix}__learning_rate_xgb_{copy_i}',-4, -1),
                              f'{method_prefix}__subsample': hp.uniform(f'{method_prefix}__subsample_xgb_{copy_i}',0.5,1.0),
                              f'{method_prefix}__colsample_bytree': hp.uniform(f'{method_prefix}__colsample_bytree_xgb_{copy_i}',0.6,1.0)
                            }
                clf_list.append(m_name)
        else:
            for copy_i in range(x):
                m_name=f'xgb_{copy_i}'
                method_archive.methods[m_name]={f'{method_prefix}': hp.choice(f'{method_prefix}_xgb_{copy_i}',[xgb.XGBClassifier(objective ='binary:logistic', verbosity =1,nthread=xgb_threads, random_state=random_state[copy_i] )]),
                              f'{method_prefix}__eval_metric':hp.choice(f'{method_prefix}__eval_metric_xgb_{copy_i}', ['auc','mlogloss']),
                              f'{method_prefix}__n_estimators':hp.choice(f'{method_prefix}__n_estimators_xgb_{copy_i}', n_estimators),
                              f'{method_prefix}__max_depth':hp.choice(f'{method_prefix}__max_depth_xgb_{copy_i}', range(3,5)),
                              f'{method_prefix}__min_child_weight': hp.choice(f'{method_prefix}__min_child_weight_xgb_{copy_i}',range(2, 14, 3)),
                              f'{method_prefix}__gamma': hp.choice(f'{method_prefix}__gamma_xgb_{copy_i}',[0.5, 1, 1.5, 2, 5]),
                              f'{method_prefix}__learning_rate': hp.choice(f'{method_prefix}__learning_rate_xgb_{copy_i}',np.logspace(-4, -1, 4)),
                              f'{method_prefix}__subsample': hp.choice(f'{method_prefix}__subsample_xgb_{copy_i}',[0.5,0.8,1.0]),
                              f'{method_prefix}__colsample_bytree': hp.choice(f'{method_prefix}__colsample_bytree_xgb_{copy_i}',[0.6,0.8,1.0])
                        }
                clf_list.append(m_name)
    return method_archive,clf_list

def add_lgbm_xtimes_hyperopt(method_archive,method_prefix,x,distribution_defaults=False,random_state=None,xgb_threads=4,n_estimators=None,regressor=False):
    """
    duplicates lgbm x times with different random_states initializations and returns the corresponding dictionary keys

    Args:
         method_archive: the archive class holding the method dictionary
         method_prefix: the method prefix string
         x: number of copies
         distribution_defaults: set to true if you want to use distribution defaults
         random_state: random state initialisation, if value and not a list, a list is generated from this value
         xgb_threads: number of threads used in lgbm
         n_estimator: param list of number of estimators
         regressor: boolean to be set to add regressor
    """
    if n_estimators is None : n_estimators=[100,250,500,1000]           
    if random_state is not None and not isinstance(random_state, (list)):
        random_state=[(random_state+(7+random_state)**i)%31393 for i in range(x)]
    if random_state is None:
        random_state=[None for i in range(x)]
    assert len(random_state)==x, 'len of random states is not equal to number of method duplicates'
    clf_list=[]
    if regressor:
        if distribution_defaults:
            for copy_i in range(x):
                m_name=f'lgbm_{copy_i}'
                method_archive.methods[m_name]={f'{method_prefix}': hp.choice(f'{method_prefix}_lgbm_{copy_i}',[LGBMRegressor(n_jobs=xgb_threads,verbosity=-1, random_state=random_state[copy_i])]),
                              f'{method_prefix}__n_estimators':hp.choice(f'{method_prefix}_lgbm__n_estimators_{copy_i}', n_estimators),
                              f'{method_prefix}__max_depth': hp.choice(f'{method_prefix}_lgbm__max_depth_{copy_i}',range(3,8)),
                              f'{method_prefix}__reg_alpha': hp.loguniform(f'{method_prefix}_lgbm__reg_alpha_{copy_i}',-4, 1),
                              f'{method_prefix}__reg_lambda': hp.loguniform(f'{method_prefix}_lgbm__reg_lambda_{copy_i}',-4, 1),
                              f'{method_prefix}__min_child_samples': hp.choice(f'{method_prefix}_lgbm__min_child_samples_{copy_i}',range(2, 14, 3)),
                              f'{method_prefix}__subsample': hp.choice(f'{method_prefix}_lgbm__subsample_{copy_i}',[0.5,0.8,1.0]),
                              f'{method_prefix}__learning_rate': hp.loguniform(f'{method_prefix}_lgbm__learning_rate_{copy_i}',-4, -1),
                              f'{method_prefix}__num_leaves': hp.choice(f'{method_prefix}_lgbm__num_leaves_{copy_i}',[2,5,10,15,20,30]),
                              f'{method_prefix}__boosting_type':hp.choice(f'{method_prefix}_lgbm__boosting_type_{copy_i}',['gbdt','dart'])
                            }
                clf_list.append(m_name)
        else:
            for copy_i in range(x):
                m_name=f'lgbm_{copy_i}'
                method_archive.methods[m_name]={f'{method_prefix}': hp.choice(f'{method_prefix}_lgbm_{copy_i}',[LGBMRegressor(n_jobs=xgb_threads,verbosity=-1, random_state=random_state[copy_i])]),
                              f'{method_prefix}__n_estimators':hp.choice(f'{method_prefix}_lgbm__n_estimators_{copy_i}', n_estimators),
                              f'{method_prefix}__max_depth': hp.choice(f'{method_prefix}_lgbm__max_depth_{copy_i}',range(3,8)),
                              f'{method_prefix}__reg_alpha': hp.choice(f'{method_prefix}_lgbm__reg_alpha_{copy_i}',np.logspace(-4, -1, 4)),
                              f'{method_prefix}__reg_lambda': hp.choice(f'{method_prefix}_lgbm__reg_lambda_{copy_i}',np.logspace(-4, -1, 4)),
                              f'{method_prefix}__min_child_samples': hp.choice(f'{method_prefix}_lgbm__min_child_samples_{copy_i}',range(2, 14, 3)),
                              f'{method_prefix}__subsample': hp.choice(f'{method_prefix}_lgbm__subsample_{copy_i}',[0.5,0.8,1.0]),
                              f'{method_prefix}__learning_rate': hp.choice(f'{method_prefix}_lgbm__learning_rate_{copy_i}',np.logspace(-4, -1, 4)),
                              f'{method_prefix}__num_leaves': hp.choice(f'{method_prefix}_lgbm__num_leaves_{copy_i}',[2,5,10,15,20,30]),
                              f'{method_prefix}__boosting_type':hp.choice(f'{method_prefix}_lgbm__boosting_type_{copy_i}',['gbdt','dart'])
                            }
                clf_list.append(m_name)
    else:
        if distribution_defaults:
            for copy_i in range(x):
                m_name=f'lgbm_{copy_i}'
                method_archive.methods[m_name]={f'{method_prefix}': hp.choice(f'{method_prefix}_lgbm_{copy_i}',[LGBMClassifier(n_jobs=xgb_threads,verbosity=-1, random_state=random_state[copy_i])]),
                              f'{method_prefix}__n_estimators':hp.choice(f'{method_prefix}_lgbm__n_estimators_{copy_i}', n_estimators),
                              f'{method_prefix}__max_depth': hp.choice(f'{method_prefix}_lgbm__max_depth_{copy_i}',range(3,8)),
                              f'{method_prefix}__reg_alpha': hp.loguniform(f'{method_prefix}_lgbm__reg_alpha_{copy_i}',-4, 1),
                              f'{method_prefix}__reg_lambda': hp.loguniform(f'{method_prefix}_lgbm__reg_lambda_{copy_i}',-4, 1),
                              f'{method_prefix}__min_child_samples': hp.choice(f'{method_prefix}_lgbm__min_child_samples_{copy_i}',range(2, 14, 3)),
                              f'{method_prefix}__subsample': hp.choice(f'{method_prefix}_lgbm__subsample_{copy_i}',[0.5,0.8,1.0]),
                              f'{method_prefix}__learning_rate': hp.loguniform(f'{method_prefix}_lgbm__learning_rate_{copy_i}',-4, -1),
                              f'{method_prefix}__num_leaves': hp.choice(f'{method_prefix}_lgbm__num_leaves_{copy_i}',[2,5,10,15,20,30]),
                              f'{method_prefix}__boosting_type':hp.choice(f'{method_prefix}_lgbm__boosting_type_{copy_i}',['gbdt','dart'])
                            }
                clf_list.append(m_name)
        else:
            for copy_i in range(x):
                m_name=f'lgbm_{copy_i}'
                method_archive.methods[m_name]={f'{method_prefix}': hp.choice(f'{method_prefix}_lgbm_{copy_i}',[LGBMClassifier(n_jobs=xgb_threads,verbosity=-1, random_state=random_state[copy_i])]),
                              f'{method_prefix}__n_estimators':hp.choice(f'{method_prefix}_lgbm__n_estimators_{copy_i}', n_estimators),
                              f'{method_prefix}__max_depth': hp.choice(f'{method_prefix}_lgbm__max_depth_{copy_i}',range(3,8)),
                              f'{method_prefix}__reg_alpha': hp.choice(f'{method_prefix}_lgbm__reg_alpha_{copy_i}',np.logspace(-4, -1, 4)),
                              f'{method_prefix}__reg_lambda': hp.choice(f'{method_prefix}_lgbm__reg_lambda_{copy_i}',np.logspace(-4, -1, 4)),
                              f'{method_prefix}__min_child_samples': hp.choice(f'{method_prefix}_lgbm__min_child_samples_{copy_i}',range(2, 14, 3)),
                              f'{method_prefix}__subsample': hp.choice(f'{method_prefix}_lgbm__subsample_{copy_i}',[0.5,0.8,1.0]),
                              f'{method_prefix}__learning_rate': hp.choice(f'{method_prefix}_lgbm__learning_rate_{copy_i}',np.logspace(-4, -1, 4)),
                              f'{method_prefix}__num_leaves': hp.choice(f'{method_prefix}_lgbm__num_leaves_{copy_i}',[2,5,10,15,20,30]),
                              f'{method_prefix}__boosting_type':hp.choice(f'{method_prefix}_lgbm__boosting_type_{copy_i}',['gbdt','dart'])
                            }
                clf_list.append(m_name)
    return method_archive,clf_list

def retrieve_search_options(search_type='grid',use_distributions=False, n_iter=None):
    """
    set the optimization options accordingly to the provide search_type
    
    Args:
         search_type: string representing the type of optimization ['grid','randomized','hyperopt']
         use_distributions: boolean to use distributions instead of discrete values for the parameters
         n_iter: number of iterations [=None]
    
    Returns:
        tuple of n_iter, boolean indicating hyperopt, boolean indicating use of distributions
    """
    assert search_type in ['grid','randomized','hyperopt']
    hyperopt_defaults=False
    if search_type =='randomized':
        if n_iter is None: n_iter=100
    elif search_type =='hyperopt':
        hyperopt_defaults=True
        if n_iter is None: n_iter=100
    else: 
        use_distributions=False
        n_iter=None
    return n_iter, hyperopt_defaults, use_distributions

from typing import List

def get_clustering_algorithm(clustering: str='Bottleneck', * ,
                             n_clusters:int=20,
                             cutoff:float=0.6,
                             include_chirality=False,
                             verbose:bool=0,
                             random_state:int=42,
                             radius:int=2,
                             nBits:int=1024,
                             feature_generators: dict= {},
                             used_features:List[str]=None,
                             butina_feature_gen: FeatureGenerator=None):
    """
     Helper function to retrieve corresponding clustering algorithm based on provided input string and arguments
     
     Args:
         clustering [str]: strings to select clustering algorithm
                     - Bottleneck: kmeans on provided feature generators
                     - Butina: butina clustering with reassignment
                     - HierarchicalButina: Hierarchical butina clustering with reassignment
                     - Scaffold: MurckoScaffold Clustering
        n_cluster [int]: number of cluster for kmeans (clustering=Bottleneck)
        cutoff [float]: cutoff for butina or hierarchical butina
        include_chiralty: chiralty option for MurckoScaffolds
        verbose: boolean to indicate more print statements
        random_state: seed for random number generators
        radius: radius for ecfp generation for butina clusterings
        nBits: number of features for ecfp generation for butina clustering
        feature_generators: dictionary of AutoMoL feature generators used for kmeans (clustering=Bottleneck)
        used_features: list of keys from the feature_generators to select used feature generators for kmeans (clustering=Bottleneck)
        butina_feature_gen: provided feature generator for butina clustering must have bitVect return options, see ECFPGenerator
        
    Returns:
        a AutoMoL ClusteringAlgorithm
    """
    assert clustering in ['Bottleneck','Butina','HierarchicalButina', 'Scaffold'], 'provide one of Bottleneck or Butina or Scaffold'
    clust_algo=None
    if clustering== 'Scaffold':
        clust_algo=MurckoScaffoldClustering(include_chirality=include_chirality)
    elif clustering== 'Butina':
        if butina_feature_gen is None:
            butina_feature_gen=ECFPGenerator(radius=radius, nBits =nBits)
        clust_algo=ButinaSplitReassigned(cutoff = cutoff,feature_generator=butina_feature_gen)
    elif clustering=='HierarchicalButina':
        if butina_feature_gen is None:
            butina_feature_gen=ECFPGenerator(radius=radius, nBits =nBits)
        if cutoff is not None and not isinstance(cutoff, (list)):
            cutoff=[cutoff, np.minimum(cutoff+0.1,np.maximum(cutoff,0.9))]
        clust_algo=HierarchicalButina(cutoff = cutoff,feature_generator=butina_feature_gen)
    elif clustering== 'Bottleneck':
        clust_algo=KmeansForSmiles(n_groups=n_clusters,feature_generators=feature_generators,used_features=used_features,random_state=random_state)

    return clust_algo

def print_available_keys(task='Regression',hyperopt_defaults=False, distribution_defaults=False,prefixes=None):
    """
    prints the available method keys
    
    Args:
         task: string of the task [Regression, RegressionClassification, Classification]
         hyperopt_defaults: boolean indicating hyperopt
         distribution_defaults: boolean for distributions'
         prefixes: dictionary with prefixes
    """
    clf=False
    if task=='Classification' or task=='clf':
        clf=True
        if prefixes is None: prefixes={'method_prefix':'clf',
                                       'dim_prefix':'reduce_dim',
                                       'estimator_prefix':'est_pipe'}
        method_archive=ClassifierArchive(method_prefix=prefixes['method_prefix'], distribution_defaults=distribution_defaults, hyperopt_defaults=hyperopt_defaults, random_state=3,xgb_threads=3) 
    else:
        if prefixes is None: prefixes={'method_prefix':'reg',
                                      'dim_prefix':'reduce_dim',
                                      'estimator_prefix':'est_pipe'}
        method_archive=RegressorArchive(method_prefix=prefixes['method_prefix'], distribution_defaults=distribution_defaults, hyperopt_defaults=hyperopt_defaults, random_state=3,xgb_threads=3)
        
    dim_archive=ReducedimArchive(method_prefix=prefixes['dim_prefix'], distribution_defaults=distribution_defaults, hyperopt_defaults=hyperopt_defaults, random_state=3,clf=clf)
    if hyperopt_defaults:
        print('Output may appear strange due to internal representation of hyperopt choice for the method')
    print('*****************************************\nkey: Estimator\n*****************************************')
    for key,method in method_archive.get_all_method_keys_plus_estimators():
        print(f'\33[1m{key}\33[0m: {method}')
    print('*****************************************\nKey: Dimensionality reduction method\n*****************************************')
    for key,method in dim_archive.get_all_method_keys_plus_estimators():
        print(f'\33[1m{key}\33[0m: {method}')


class ModelAndParams:
    """
    A class used to create the underlying model and methods and grid parameters. 
    
    There are a few minimal parameters to be set, resulting in a few pre set defaults and yet full flexibility is also provided if required
    """
    
    def __init__(self,model='stereo_SMILES', task='Regression',computional_load='cheap', labelnames=None,
                 distribution_defaults=False, hyperopt_defaults=None ,use_gpu=False, normalizer=True,
                 top_normalizer=True, random_state=42, red_dim_list=None, method_list=None,
                 blender_list=None, model_config=None, force_gridsearch=False, randomized_iterations=None,
                 method_archive=None, dim_archive=None, blender_archive=None, prefixes=None, verbose=False,
                 hyperopt_threads=None, n_jobs=None, xgb_threads=None, rfr_threads=None,
                 n_jobs_dict=None, compute_SD=True, use_sample_weight=False,feature_generators=None,
                 local_dim_red=False,dim_red_n_components=None, relative_modelling=False,
                 feature_operation:str='concat',property_operation:str='minus'
                 ):
        """
        user_interface function that returns the correct stacking_model, methods and parameters based on a few user set variables while maintaining full flexibility by checking input parameters.
        
        The minimal parameters are: model, task, computional load and labelnames. Using the previous minimal parameters will pick defaults and you will able to run your classifier/regressor. You can further customize your classifier/regressor by using the other options.

        
        Args:
             model: String to choose between models, only choice for now is 'CHEMBL'
             task: String to choose between Classification of Regression
             computational_load: String to choose between 'cheap', 'moderate' or 'expensive' wich increasing computational budget and likely but not necessarily better results
             labelnames: dictionary containing the names of the classes for classification
             distribution_defaults: boolean to use default distributions as estimator parameter options (instead of discrete values) for randomized_search
             hyperopt_defaults: boolean to set if use of hyperopt optimization is prefered for parameter optimization
             use_gpu: boolean to indicate use of gpu
             normalizer: boolean to add StandardScaler to base estimator pipeline
             top_normalizer: boolean to add StandardScaler to top estimator (or blender or final estimator) pipeline
             random_state: set to value for reproducable results
             red_dim_list: list of keys to retrieve dimensionality reduction methods
             method_list: list of keys to retrieve base estimator methods
             blender_list: list of keys to retrieve top estimator methods
             model_config: String to select the specific model configuration/hierarchy
                    model_config='inner_methods': inner folds are used to find models. The predictions of these models are average for regression or given to a voting classifier for classification
                    model_config='inner_stacking': inner folds are used to find scikit stacking models (#stacking models = #outer folds)
                    model_config='single_stack': outer folds are used to find one scikit stacking model
                    model_config='top_method': builds on 'inner_methods' but the predictions of these base estimators are given to a top estimator or blender
                    model_config='top_stacking': builds on 'inner_methods' but the predictions of these base estimators are given a scikit stacking model
                    model_config='single_method': fits a single method on the outer folds
                    model_config='stacking_stacking': [only for classification] builds on 'inner_stacking' but the predictions of these stacking models are given a scikit stacking model
             force_gridsearch: boolean to force the use of gridsearch instead of randomized search
             randomized_iterations: unsigned integer to set the number of iterations in randomized search
             method_archive: a derived MethodArchive object to retrieve to classifiers/regressors for the base estimators
             dim_archive: a derived MethodArchive object to retrieve to dimensionality reduction methods
             blender_archive: a derived MethodArchive object to retrieve to classifiers/regressors for the top estimator
             prefixes: dictionary holding the different prefixes used in the pipelines for creating the correct parameter options list (of lists) (default:None, e.g classifier: {'method_prefix':'clf','dim_prefix':'reduce_dim','estimator_prefix':'est_pipe'} and regressor: {'method_prefix':'reg','dim_prefix':'reduce_dim','estimator_prefix':'est_pipe'})
             verbose: set for prints
             hyperopt_threads: the number of threads used when using hyperopt search (uses pyspark)
             n_jobs: number of jobs for sklearn param search
             xgb_threads: number of threads used in xgboost or lgbm (ensemble trees)
             compute_SD: boolean to add std to the output (default:True)
             use_sample_weight: boolean to indicate use of sample_weights
             n_jobs_dict: Dictionary with the number number of threads for each level ('outer_jobs', 'inner_jobs' and 'method_jobs')
             feature_generators: Dictionary of feature generators
             local_dim_red: boolean to indicate featurewise dimensionality reduction
             dim_red_n_components: list of number features to reduce the data too
        """
        assert not ((hyperopt_defaults or distribution_defaults) and force_gridsearch), 'It is not possible to force gridsearch and using hyperopt or distributions default'
        n_iter=None
        clf=False


        if hyperopt_defaults is None:
            if computional_load=='expensive':
                hyperopt_defaults=True
            else:
                hyperopt_defaults=False
                
        if n_jobs_dict is None:
            n_jobs_dict={'outer_jobs':None,
                            'inner_jobs':-1,
                            'method_jobs':2}
        if n_jobs is not None:
            n_jobs_dict['inner_jobs']=n_jobs
        if hyperopt_defaults and hyperopt_threads is not None:
            n_jobs_dict['inner_jobs']=hyperopt_threads

        if xgb_threads is not None:
            n_jobs_dict['method_jobs']=xgb_threads
        elif rfr_threads is not None:
            n_jobs_dict['method_jobs']=rfr_threads
        
        if rfr_threads is None:
            rfr_threads=n_jobs_dict['method_jobs']
        if xgb_threads is None:
            xgb_threads=n_jobs_dict['method_jobs']
            
        regressionclassifier=False

        if task.lower()=='classification' or task.lower()=='clf':
            clf=True
            if prefixes is None: prefixes={'method_prefix':'clf',
                                           'dim_prefix':'reduce_dim',
                                           'estimator_prefix':'est_pipe'}
            if method_archive is None: 
                method_archive=ClassifierArchive(method_prefix=prefixes['method_prefix'], distribution_defaults=distribution_defaults, hyperopt_defaults=hyperopt_defaults, random_state=random_state, xgb_threads=xgb_threads, rfr_threads=rfr_threads,method_jobs=n_jobs_dict['method_jobs']) 

            if computional_load=='cheap':
                if model_config is None: model_config='top_method'
                if red_dim_list is None: red_dim_list=['passthrough']
                if method_list is None: 
                    method_list=['lr' ,'SVC','knn']
                    if use_sample_weight:
                        method_list=['lr' ,'SVC']
                if blender_list is None: blender_list=['lr']
            elif computional_load=='expensive':
                if model_config is None: model_config='top_stacking'
                n_iter=100
                if red_dim_list is None: red_dim_list=['passthrough']
                if method_list is None:
                    if hyperopt_defaults:
                        method_archive,clf_list=add_lgbm_xtimes_hyperopt(method_archive,prefixes['method_prefix'],3,
                                                         xgb_threads=xgb_threads, distribution_defaults=distribution_defaults,
                                                         random_state=random_state[0],regressor=False)
                    else:
                        clf_list=method_archive.duplicate_method_xtimes(method_name='lgbm',x=3,random_state=random_state[0])
                    method_list=['lr','knn' ,'SVC','sgdc','dtc','lgbm']+clf_list
                if blender_list is None: blender_list=['lr','lgbm']

            else:
                if model_config is None: model_config='top_method'
                n_iter=100
                if red_dim_list is None: red_dim_list=['passthrough']
                if method_list is None: 
                    method_list=['lr','knn' ,'SVC','sgdc','lgbm']
                    if use_sample_weight:
                        method_list=['lr' ,'SVC','sgdc','lgbm']
                if blender_list is None: blender_list=['lr','lgbm']    

        else:

            if task.lower()=='regressionclassification' or task.lower()=='regclf':
                regressionclassifier=True

            if prefixes is None: prefixes={'method_prefix':'reg',
                                          'dim_prefix':'reduce_dim',
                                          'estimator_prefix':'est_pipe'}
            if method_archive is None: 
                method_archive=RegressorArchive(method_prefix=prefixes['method_prefix'], distribution_defaults=distribution_defaults, hyperopt_defaults=hyperopt_defaults, random_state=random_state, xgb_threads=xgb_threads, rfr_threads=rfr_threads,method_jobs=n_jobs_dict['method_jobs'])

            if computional_load=='cheap':
                if model_config is None: model_config='top_method'
                if red_dim_list is None: red_dim_list=['passthrough']
                if method_list is None: 
                    method_list=['svr','lasso','kernelridge','pls' ]
                    if use_sample_weight:
                        method_list=['svr','lasso','kernelridge' ]
                if blender_list is None: blender_list=['svr','lasso','kernelridge']
            elif computional_load=='expensive':
                if model_config is None: model_config='top_stacking'
                n_iter=100
                if red_dim_list is None: red_dim_list=['passthrough']
                if method_list is None:
                    if hyperopt_defaults:
                        method_archive,reg_list=add_lgbm_xtimes_hyperopt(method_archive,prefixes['method_prefix'],3,
                                                         xgb_threads=xgb_threads, distribution_defaults=distribution_defaults,
                                                         random_state=random_state[0],regressor=True)
                    else:
                        reg_list=method_archive.duplicate_method_xtimes(method_name='lgbm',x=3,random_state=random_state[0])
                    method_list=[ 'svr','lasso','kernelridge','sgdr','dtr','lgbm']+reg_list
                if blender_list is None: 
                    blender_list=['svr','lasso','kernelridge','lgbm']

            else:
                if model_config is None: model_config='top_method'
                if red_dim_list is None: red_dim_list=['passthrough']
                n_iter=100
                if method_list is None: 
                    method_list=['pls' , 'svr','lasso','kernelridge','sgdr','dtr','lgbm']
                    if use_sample_weight:
                        method_list=['svr','lasso','kernelridge','sgdr','dtr','lgbm']
                if blender_list is None: blender_list=['svr','lasso','kernelridge','lgbm']



        if randomized_iterations is not None: n_iter=randomized_iterations
        if force_gridsearch: n_iter=None

        if n_iter is None:
            paramsearch=GridSearch(refit=True,n_jobs=n_jobs_dict['inner_jobs'], verbose=verbose)
        else:
            if hyperopt_defaults:
                paramsearch=HyperoptSearch(n_iter=n_iter,n_jobs=n_jobs_dict['inner_jobs'])
            else:
                paramsearch=RandomizedSearch(refit=True,n_jobs=n_jobs_dict['inner_jobs'], verbose=verbose,n_iter=n_iter)


        if dim_archive is None: dim_archive=ReducedimArchive(method_prefix=prefixes['dim_prefix'], distribution_defaults=distribution_defaults, hyperopt_defaults=hyperopt_defaults, random_state=random_state,dim_red_n_components=dim_red_n_components)
        if blender_archive is None: blender_archive=method_archive

        #get stacking model based on the given config string
        self.model_selector=MethodConfigurations(classifier=clf,distribution_defaults=distribution_defaults, hyperopt_defaults=hyperopt_defaults, model_config=model_config, regressionclassifier=regressionclassifier,n_jobs_dict=n_jobs_dict)    
        
        self.model=model
        self.use_gpu=use_gpu
        self.compute_SD=compute_SD
        self.labelnames=labelnames
        self.feature_generators=feature_generators
        self.method_list=method_list
        self.red_dim_list=red_dim_list
        self.blender_list=blender_list
        self.method_archive=method_archive
        self.dim_archive=dim_archive
        self.blender_archive=blender_archive
        self.prefixes=prefixes
        self.normalizer=normalizer
        self.top_normalizer=top_normalizer
        self.random_state=random_state
        self.local_dim_red=local_dim_red
        self.paramsearch=paramsearch
        self.verbose=verbose
        self.relative_modelling=relative_modelling
        self.feature_operation=feature_operation
        self.property_operation=property_operation
        
    def get_model_and_params(self):
        """
        returns the model and different grid parameters
        
        Returns: 
            tuple of stacking model creator, dictionary of prefixes, grid parameters for the base estimator, grid parameters for the final estimator and ParamSearch object 
        """
        stacked_model=self.model_selector.get_stacking_model(model=self.model,use_gpu=self.use_gpu,compute_SD=self.compute_SD, labelnames=self.labelnames, feature_generators=self.feature_generators,
                                                            verbose=self.verbose, relative_modelling=self.relative_modelling,
                                                            feature_operation= self.feature_operation, property_operation= self.property_operation)

        prefixes,params_grid,blender_params=self.model_selector.get_cv_params(method_list=self.method_list, dim_list=self.red_dim_list, blender_list=self.blender_list,
                                                        method_archive=self.method_archive, dim_archive=self.dim_archive, blender_archive=self.blender_archive,
                                                        prefixes=self.prefixes, normalizer=self.normalizer, top_normalizer=self.top_normalizer,
                                                        random_state=self.random_state,local_dim_red=self.local_dim_red)

        return stacked_model,prefixes,params_grid,blender_params,self.paramsearch

    def get_feature_params(self,selected_features=None,dim_list=None):
        """
        retrieve the grid parameters based on the selected features
        
        Args:
             selected_features: list of keys representing the feature generators
             dim_list: list of dimensionality reduction method keys [=None], if None default are used 
        
        Returns: 
            tuple of base_estimator grid parameters and final estimator grid parameters
        """
        if dim_list is None:
            dim_list=self.red_dim_list
        prefixes,params_grid,blender_params=self.model_selector.get_cv_params(method_list=self.method_list, dim_list=dim_list, blender_list=self.blender_list,
                                                        method_archive=self.method_archive, dim_archive=self.dim_archive, blender_archive=self.blender_archive,
                                                        prefixes=self.prefixes, normalizer=self.normalizer, top_normalizer=self.top_normalizer,
                                                        random_state=self.random_state,local_dim_red=self.local_dim_red, used_features=selected_features)

        return params_grid,blender_params

    
        


class ResultDesigner:
    """
    displays the different matplotlib figures
    """
    
    
    def __init__(self,line_width=2,fig_per_row=3,colors=None):
        """
        Initialization to set line width, colors and nb of figs per row for the figures in the generated results
        
        Args:
             line_width:
             fig_per_row: nb of figs per row for matplotlib subfigures configuration
             colors: list of color strings
        """
        if colors is None:
            self.colors=["darkorange","darkred","darkgreen",'deepskyblue','darkviolet','orangered','yellowgreen']
        else: 
            self.colors=colors
        self.fig_per_row=fig_per_row
        self.line_width=line_width
            
            
    def show_classification_report(self,properties,out,y_true,labelnames,cmap='PiYG',stream_lit=False ):
        """
        prints the classification results: scikit classification report, ROC-AUC plot, confusion matrix and Recall vs threshold plot
        
        Args:
             properties: property names list present in the output
             out: model output containing the predictions
             y_true: the array of the true values
             labelnames: dictionary of class names 
             cmap: the colormap
        
        Returns: 
            dictionary with youden-j values
        """
        if not isinstance(properties, (list)):
            properties=[properties]
        if not isinstance(cmap, (list)):
            cmap=[cmap,cmap]
        
        nb_props=len(properties)

        Youden_dict={}
        
        fig, axs = plt.subplots(len(properties), 3, figsize=(18 ,6*len(properties)))
        i=0
        for ip,p in enumerate(properties):
            print("####################################################")
            print(p)
            y_pred=out[f'predicted_{p}']
            nb_classes=len(list(labelnames[p].values()))
            y_pred_proba=np.concatenate(tuple(np.expand_dims(out[f'predicted_proba_{p}_class_{labelnames[p][c]}'],axis=1) for c in range(nb_classes)), axis=1)
            #y_pred_proba=out[f'predicted_proba_{p}']
            mask=np.isnan(y_true[ip])| np.isnan(y_pred.astype(float))
            
            ax=axs.flat[i]    
            plot_clf_matrix(y_pred[~mask].astype(int), y_true[ip][~mask],title=p, ax=ax,labels=[index for index in range(y_pred_proba.shape[1])],prop_labels=labelnames[p],cmap=cmap[0] )
            i+=1
            
            ax=axs.flat[i]
            Youden_th, Youden_val = plot_clf_auc(y_pred_proba[~mask,:], y_true[ip][~mask],title=p, ax=ax,colors=self.colors, line_width=self.line_width, prop_labels=labelnames[p] )
            Youden_dict[p]=Youden_th
            Youden_dict[f'val_{p}']=Youden_val
            i+=1
            
            ax=axs.flat[i]
            plot_clf_confusion(y_pred[~mask].astype(int), y_true[ip][~mask],title=p, ax=ax,prop_labels=labelnames[p],cmap=cmap[1] )
            i+=1
            
        fig.tight_layout()
        fig.show()
        if stream_lit:
            return Youden_dict,fig
        else:
            return Youden_dict
    
    
    def show_clf_threshold_report(self,properties,out,y_true,labelnames,youden_dict=None,stream_lit=False ):
        """
        displays the different threshold tuning figures
        
        Args:
             properties: the properties
             out: the generated output
             y_true: the true values
             labelnames: labelnames of the classes
             youden_dict: dictionary containing the optimal youden-J statistic
             streamlit: boolean to return dictionary of figures
        
        Returns: 
            dictionary with f1 values
        """
        if not isinstance(properties, (list)):
            properties=[properties]
        
        nb_props=len(properties)
        figures={}
        F1_dict={}
        i=0
        for ip,p in enumerate(properties):
            print("####################################################")
            print(p)
            y_pred=out[f'predicted_{p}']
            nb_classes=len(list(labelnames[p].values()))
            y_pred_proba=np.concatenate(tuple(np.expand_dims(out[f'predicted_proba_{p}_class_{labelnames[p][c]}'],axis=1) for c in range(nb_classes)), axis=1)
            #y_pred_proba=out[f'predicted_proba_{p}']
            mask=np.isnan(y_true[ip]) | np.isnan(y_pred.astype(float))
            
            fig, axs = plt.subplots(2, nb_classes, figsize=(6*nb_classes ,9))
            i=0
            F1_th=[]
            F1_val=[]
            for c in range(nb_classes):
                ax=axs.flat[i]
                F1_th_c, F1_val_c = plot_clf_f1(y_pred_proba[~mask,:], y_true[ip][~mask],title=f'Threshold tuning {p}', ax=ax,colors=self.colors,line_width=self.line_width,c=c, labelname=labelnames[p][c],youden=youden_dict[p][c],youden_val=youden_dict[f'val_{p}'][c] )
                F1_th.append(F1_th_c)
                F1_val.append(F1_val_c)
                i+=1
            for c in range(nb_classes):
                ax=axs.flat[i]
                plot_bar={}
                plot_bar[f'Pred. Prob. {labelnames[p][c]}_{p}']=y_pred_proba[~mask,c]
                plot_bar[f'categories_{p}']=y_true[ip][~mask]
                plot_confusion_bars_from_categories(pd.DataFrame.from_dict(plot_bar),pro1= f'Pred. Prob. {labelnames[p][c]}_{p}',pro2= f'categories_{p}', 
                        bins1=np.linspace(0,1,11)[1:-1],
                        ax=ax,title=f'Pred. prob. {labelnames[p][c]} for {p}', figsize=(5,5),
                        leg_title=f'Pred. Prob. {labelnames[p][c]}',x_title=f'True class',
                        labelnames=labelnames[p]
                )
                i+=1
                
            F1_dict[p]=F1_th
            F1_dict[f'f1_{p}']=F1_val
            fig.tight_layout()
            fig.show()
            figures[p]=fig
        if stream_lit:
            return F1_dict,figures
        else:
            return F1_dict
        
    def show_regression_report(self,properties,out,y_true,mask_scatter=False,prop_cliffs=None, leave_grp_out=None,positive_cls="<", bins=50, bin_window_average=False):
        """       
        Shows the regression results
        
        Args:
             properties: property names list present in the output
             out: model output containing the predictions
             y_true: the array of the true values
             mask_scatter: mask scatter
             prop_cliffs: dictionary of property cliffs
             leave_grp_out: array of leave group out indices
             positive_cls: indicate positive class as > or < 
             bins: number of bins (or window for moving average)
             bin_window_average: used binned average instead of moving average
        """
        if not isinstance(properties, list):
            properties=[properties]
        if prop_cliffs is None:
            prop_cliffs={p:None for p in properties}
            
        
        plt.close('all')
        fig, axs = plt.subplots(len(properties), 3, figsize=(21 ,6*len(properties)))
        i=0
        for ip,p in enumerate(properties):
            #fig, axs = plt.subplots(1, 2, figsize=(16 ,6))
            y_pred=out[f'predicted_{p}']
            assert len(y_true[ip])>1, f'Length of provided true {p}-values is zero'
            assert len(y_pred)>1, f'Length of predicted {p}-values is zero'
            mask=np.isnan(y_true[ip])| np.isnan(y_pred.astype(float))
            non_nan_indices=np.cumsum(~mask)-1
            fig.subplots_adjust(top=0.82)
            prop_cliff_indices=None
            lgo_indices=None
            if prop_cliffs[p] is not None:
                prop_cliff_indices=non_nan_indices[prop_cliffs[p]]
                if len(prop_cliff_indices)==0: prop_cliff_indices=None
            if leave_grp_out is not None:
                lgo_indices=non_nan_indices[leave_grp_out]
                if len(lgo_indices)==0: lgo_indices=None
                
            plot_reg_model_with_error(y_pred[~mask],  y_true[ip][~mask],title=f'{p}', alpha=0.25, mask_scatter=mask_scatter, bins=bins, ax=axs.flat[i], prop_cliffs=prop_cliff_indices, leave_grp_out=lgo_indices,bin_window_average=bin_window_average)
            i+=1
            plot_bar={}
            plot_bar[f'predicted_{p}']=y_pred[~mask]
            plot_bar[f'abs_error_{p}']=np.absolute( y_true[ip][~mask]-y_pred[~mask])
            plot_confusion_bars_from_continuos(pd.DataFrame.from_dict(plot_bar),pro1= f'predicted_{p}',pro2= f'abs_error_{p}', 
                    bins2=np.linspace(np.min(plot_bar[f'abs_error_{p}']), np.max(plot_bar[f'abs_error_{p}']),6)[1:-1],
                    bins1=np.linspace(np.min(plot_bar[f'predicted_{p}']), np.max(plot_bar[f'predicted_{p}']),5)[1:-1],
                    ax=axs.flat[i],title=f'{p}', figsize=(5,5),leg_title='abs err'
            )
            i+=1
            
            plt.sca(axs.flat[i])
            b=np.quantile(y_true[ip][~mask], 0.001)
            e=np.quantile(y_true[ip][~mask], 0.95)
            plot_acc_pre_for_reg(y_true[ip],y_pred,title=f'{p}',b=b,e=e, good_class= positive_cls)
            i+=1

        plt.tight_layout()
        plt.show()
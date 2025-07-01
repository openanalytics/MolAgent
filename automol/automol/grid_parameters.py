"""  implementation of the different functions to generate the dictionaries for the different parameter optimizations.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""
import hyperopt
from hyperopt import hp
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from .feature_reduction import FeatureTypeDimReduction
######################################################
def make_grid_parm(normalizer=True,estimator_list=None,dim_red_list=None,dim_prefix='reduce_dim',feature_splits=None):
    """
    function that creates a list of list with all the different parameter options for the general pipeline  [normalizer, dimensionality reduction, estimator].
    
    One set of parameters consists of all the parameters for one specific method (we fit one method at the time and the method is specified in the options)
    
    Args:
        normalizer: boolean to use StandardScaler inside the pipeline
        estimator_list: list of dictionaries with the used estimators and their parameters
        dim_red_list: list of dictionaries with the used dimensionality reduction methods and their parameters
        feature_splits: A dictionary with keys {names,col_splits}, with names the names of the different feature types and col_splits, the columns indicating the start of the different features and the total at the end, e.g. [0,5,10]: 2 features types, with the first from 0 to 5 and the last from 5 to 10
    
    Returns:
        dictionary containing the methods and their parameters with the correct keys for the AutoML pipeline
    """
    parms=[]
    for estim in estimator_list:
        estim_par=[]
        if feature_splits is not None:
            nb_feat_types=len(feature_splits['names'])
            nb_dim_methods=[ len(dim_red_list[feat]) for feat in range(nb_feat_types)] 
            nb_dim_methods.append(1)
            for combi_index in range(np.prod(nb_dim_methods)):
                out={}
                if normalizer: out={**out, 'normalizer':[StandardScaler()]}
                if isinstance(estim,  type(PLSRegression())):
                    out={**out,f'{dim_prefix}__fs{index}': ['passthrough'] }
                else:
                    out={**out,f'{dim_prefix}': [FeatureTypeDimReduction(estimators=['passthrough' for m in range(nb_feat_types)], col_splits=feature_splits['col_splits'], names=feature_splits['names'] )] }
                    for index in range(nb_feat_types):
                        dim_red=dim_red_list[index][int(combi_index/(np.prod(nb_dim_methods[index+1:]))%nb_dim_methods[index])]
                        altered_dict={}
                        for key,val in dim_red.items():
                            insert_point=len(f'{dim_prefix}')
                            new_key=key[:insert_point] + f'__fs{index}' + key[insert_point:]
                            altered_dict[new_key]=val
                        out={**out,**altered_dict }
                out={**out,**estim   }
                estim_par.append(out)
        else:
            for dim_red in dim_red_list:
                out={}
                if normalizer: out={**out, 'normalizer':[StandardScaler()]}
                if isinstance(estim,  type(PLSRegression())):
                    out={**out,f'{dim_prefix}': ['passthrough'] }
                else:
                    out={**out,**dim_red }
                out={**out,**estim   }
                estim_par.append(out)
        parms.append(estim_par)
    return parms

def make_hyperopt_grid_parm(normalizer=True,estimator_list=None,dim_red_list=None,dim_prefix='reduce_dim',feature_splits=None):
    """
    function that creates a list of list with all the different parameter options in hyperopt format for the general pipeline  [normalizer, dimensionality reduction, estimator].
    
    One set of parameters consists of all the parameters for one specific method (we fit one method at the time and the method is specified in the options)
    
    Args:
        normalizer: boolean to use StandardScaler inside the pipeline
        estimator_list: list of dictionaries with the used estimators and their parameters
        dim_red_list: list of dictionaries with the used dimensionality reduction methods and their parameters
        dim_prefix: prefix to be set indicating dimensaionality reduction
        feature_splits: a dictionary with keys {names,col_splits}, with names the names of the different feature types and col_splits, the columns indicating the start of the different features and the total at the end, e.g. [0,5,10]: 2 features types, with the first from 0 to 5 and the last from 5 to 10
    
    Returns:
        dictionary containing the methods and their parameters with the correct keys for the AutoML pipeline
    """
    parms=[]
    for estim in estimator_list:
        estim_par=[]
        if feature_splits is not None:
            nb_feat_types=len(feature_splits['names'])
            nb_dim_methods=[ len(dim_red_list[feat]) for feat in range(nb_feat_types)] 
            nb_dim_methods.append(1)
            for combi_index in range(np.prod(nb_dim_methods)):
                out={}
                if normalizer: out={**out, 'normalizer':hp.choice( f'estimator{combi_index}__normalizer',[StandardScaler()])}
                if isinstance(estim,  type(PLSRegression())):
                    out={**out,f'{dim_prefix}__fs{index}': hp.choice( f'{dim_prefix}{combi_index}_pls',['passthrough'])  }
                else:
                    out={**out,f'{dim_prefix}': hp.choice( f'estimator{combi_index}__dim_red',[FeatureTypeDimReduction(estimators=['passthrough' for m in range(nb_feat_types)], col_splits=feature_splits['col_splits'], names=feature_splits['names'] )]) }
                    for index in range(nb_feat_types):
                        dim_red=dim_red_list[index][int(combi_index/(np.prod(nb_dim_methods[index+1:]))%nb_dim_methods[index])]
                        altered_dict={}
                        for key,val in dim_red.items():
                            insert_point=len(f'{dim_prefix}')
                            new_key=key[:insert_point] + f'__fs{index}' + key[insert_point:]
                            altered_dict[new_key]=val
                        out={**out,**altered_dict }
                out={**out,**estim   }
                estim_par.append(out)
        else:
            for index,dim_red in enumerate(dim_red_list):
                out={}
                if normalizer: out={**out, 'normalizer':hp.choice( f'estimator{index}__normalizer',[StandardScaler()])}
                if isinstance(estim,  type(PLSRegression())):
                    out={**out,f'{dim_prefix}': hp.choice( f'{dim_prefix}_pls',['passthrough']) }
                else:
                    out={**out,**dim_red }
                out={**out,**estim   }
                estim_par.append(out)
        parms.append(estim_par)
    return parms

###########################
def make_stacking_grid_parm(normalizer=True,blender_list=None,estimator_list=None, dim_red_list=None,dim_prefix='reduce_dim',estimator_prefix='pipe', top_normalizer=False, top_method=False, feature_splits=None):
    """
    creates list for all parameters per dimension reduction technique following scikit stacking models format with top level indicated by the prefix final_estimator and base estimators by the given estimator_prefix using the general pipeline  [normalizer, dimensionality reduction, estimator].
    
    case 1: One set of stacking parameters consists of all the parameters for the base estimator for one combination of final estimator+dimensionality reduction method (we fit one stacking model at the time). All base estimators do get the same dimensionality reduction method in the final model, further generalization is not straightforward
    case 2: base estimator and dimensionality reduction are omitted if only the parameters of the final estimator are required (set estimator_list and dim_red_list to None)
    
    Args:
        normalizer: boolean to indicate use of standardscaler for the base estimators
        blender_list: list of top estimators and parameter options in dictionary form
        estimator_list: list of base estimators and parameter options in dictionary form
        dim_red_list: list of dimensionality reduction methods and parameter options in dictionary form
        estimator_prefix: string prefix to name the pipelines of the base estimators
        dim_prefix: string prefix to name the pipelines of the dimensionality reduction methods
        top_normalizer: boolean to indicate use of standardscaler for the top estimator
        top_method: boolean to indicate use of blender instead stacking classifier
        feature_splits: a dictionary with keys {names,col_splits}, with names the names of the different feature types and col_splits, the columns indicating the start of the different features and the total at the end, e.g. [0,5,10]: 2 features types, with the first from 0 to 5 and the last from 5 to 10
    """
    #TODO? split function in two functions?
    parms=[]
    for blender in blender_list:
        if estimator_list is not None and dim_red_list is not None:
            if feature_splits is not None:
                nb_feat_types=len(feature_splits['names'])
                nb_dim_methods=[ len(dim_red_list[feat]) for feat in range(nb_feat_types)] 
                nb_dim_methods.append(1)
                for combi_index in range(np.prod(nb_dim_methods)):
                    out={}
                    for index,estim in enumerate(estimator_list):
                        if normalizer: out={**out, f'{estimator_prefix}{index}__normalizer':[StandardScaler()]}
                        out={**out,f'{estimator_prefix}{index}__{dim_prefix}': [FeatureTypeDimReduction(estimators=['passthrough' for m in range(nb_feat_types)], col_splits=feature_splits['col_splits'], names=feature_splits['names'] )] }
                        for feat_index in range(nb_feat_types):
                            dim_red=dim_red_list[feat_index][int(combi_index/(np.prod(nb_dim_methods[feat_index+1:]))%nb_dim_methods[feat_index])]
                            if isinstance(estim,  type(PLSRegression())):
                                out={**out,f'{estimator_prefix}{index}__{dim_prefix}__fs{feat_index}': ['passthrough'] }
                            else:
                                altered_dict={}
                                for key,val in dim_red.items():
                                    insert_point=len(f'{dim_prefix}')
                                    new_key=key[:insert_point] + f'__fs{feat_index}' + key[insert_point:]
                                    altered_dict[f'{estimator_prefix}{index}__{new_key}']=val
                                out={**out,**altered_dict }
                        out={**out,**{f'{estimator_prefix}{index}__{key}': value for key, value in estim.items()}  }
                    if top_normalizer: 
                        out={**out, f'final_estimator__normalizer':[StandardScaler()]}
                    else:  
                        out={**out, f'final_estimator__normalizer':['passthrough']}
                    out={**out,**{f'final_estimator__{key}': value for key, value in blender.items()}  }
                    parms.append(out)
            else:
                for dim_red in dim_red_list:
                    out={}
                    #for each base estimator a pipeline is created with name estimator_prefix+index (e.g. base_estimator1, base estimator2, ... if estimator_prefix=base estimator)
                    for index,estim in enumerate(estimator_list):
                        if normalizer: out={**out, f'{estimator_prefix}{index}__normalizer':[StandardScaler()]}
                        if isinstance(estim, type(PLSRegression())):
                            out={**out,f'{estimator_prefix}{index}__{dim_prefix}': ['passthrough'] }
                        else:
                            out={**out,**{f'{estimator_prefix}{index}__{key}': value for key, value in dim_red.items()}}
                        out={**out,**{f'{estimator_prefix}{index}__{key}': value for key, value in estim.items()}  }
                    #normalize the output of the base estimators
                    #prefix final_estimator is fixed in scikit stacking models (is possibly removed later when an estimator is used as blender instead of scikit stacking model)
                    if top_normalizer: 
                        out={**out, f'final_estimator__normalizer':[StandardScaler()]}
                    else:  
                        out={**out, f'final_estimator__normalizer':['passthrough']}
                    out={**out,**{f'final_estimator__{key}': value for key, value in blender.items()}  }
                    parms.append(out)
        else:
            out={}
            final_prefix="final_estimator__"
            if top_method:
                final_prefix=""
            if top_normalizer: 
                out={**out, f'{final_prefix}normalizer':[StandardScaler()]}
            else:  
                out={**out, f'{final_prefix}normalizer':['passthrough']}
            out={**out,**{f'{final_prefix}{key}': value for key, value in blender.items()}  }
            parms.append(out)
    return parms

def make_hyperopt_stacking_grid_parm(normalizer=True, blender_list=None,estimator_list=None, dim_red_list=None,dim_prefix='reduce_dim', estimator_prefix='pipe', top_normalizer=False, top_method=False, feature_splits=None):
    """
    creates list for all parameters in hyperopt format per dimension reduction technique following scikit stacking models format with top level indicated by the prefix final_estimator and base estimators by the given estimator_prefix using the general pipeline  [normalizer, dimensionality reduction, estimator].
    
    case 1: One set of stacking parameters consists of all the parameters for the base estimator for one combination of final estimator+dimensionality reduction method (we fit one stacking model at the time). All base estimators do get the same dimensionality reduction method in the final model, further generalization is not straightforward
    case 2: base estimator and dimensionality reduction are omitted if only the parameters of the final estimator are required (set estimator_list and dim_red_list to None)
    
    Args:
        normalizer: boolean to indicate use of standardscaler for the base estimators
        blender_list: list of top estimators and parameter options in dictionary form
        estimator_list: list of base estimators and parameter options in dictionary form
        dim_red_list: list of dimensionality reduction methods and parameter options in dictionary form
        estimator_prefix: string prefix to name the pipelines of the base estimators
        dim_prefix: string prefix to name the pipelines of the dimensionality reduction methods
        top_normalizer: boolean to indicate use of standardscaler for the top estimator
        top_method: boolean to indicate use of blender instead stacking classifier
        feature_splits: a dictionary with keys {names,col_splits}, with names the names of the different feature types and col_splits, the columns indicating the start of the different features and the total at the end, e.g. [0,5,10]: 2 features types, with the first from 0 to 5 and the last from 5 to 10
    """
    #TODO? split function in two functions?
    parms=[]
    for blender_index,blender in enumerate(blender_list):
        if estimator_list is not None and dim_red_list is not None:     
            if feature_splits is not None:
                nb_feat_types=len(feature_splits['names'])
                nb_dim_methods=[ len(dim_red_list[feat]) for feat in range(nb_feat_types)] 
                nb_dim_methods.append(1)
                for combi_index in range(np.prod(nb_dim_methods)):
                    out={}
                    for index,estim in enumerate(estimator_list):
                        if normalizer: out={**out, f'{estimator_prefix}{index}__normalizer':hp.choice( f'{estimator_prefix}{index}__estimator{combi_index}__{blender_index}___normalizer',[StandardScaler()])}
                        out={**out,f'{estimator_prefix}{index}__{dim_prefix}': hp.choice( f'{estimator_prefix}{index}__estimator{combi_index}__{blender_index}__dim_red',[FeatureTypeDimReduction(estimators=['passthrough' for m in range(nb_feat_types)], col_splits=feature_splits['col_splits'], names=feature_splits['names'] )] )}
                        for feat_index in range(nb_feat_types):
                            dim_red=dim_red_list[feat_index][int(combi_index/(np.prod(nb_dim_methods[feat_index+1:]))%nb_dim_methods[feat_index])]
                            if isinstance(estim,  type(PLSRegression())):
                                out={**out,f'{estimator_prefix}{index}__{dim_prefix}__fs{feat_index}':  hp.choice( f'{estimator_prefix}{index}__{combi_index}__{blender_index}_pls',['passthrough']) }
                            else:
                                altered_dict={}
                                for key,val in dim_red.items():
                                    insert_point=len(f'{dim_prefix}')
                                    new_key=key[:insert_point] + f'__fs{feat_index}' + key[insert_point:]
                                    altered_dict[f'{estimator_prefix}{index}__{new_key}']=val
                                out={**out,**altered_dict }
                        out={**out,**{f'{estimator_prefix}{index}__{key}': value for key, value in estim.items()}  }
                    if top_normalizer: 
                        out={**out, f'final_estimator__normalizer':hp.choice( f'final_estimator_{blender_index}_{combi_index}__normalizer',[StandardScaler()])}
                    else:  
                        out={**out, f'final_estimator__normalizer':hp.choice( f'final_estimator_{blender_index}_{combi_index}__normalizer',['passthrough'])}
                    out={**out,**{f'final_estimator__{key}': value for key, value in blender.items()}  }
                    parms.append(out)
            else:
                for dim_index,dim_red in enumerate(dim_red_list):
                    out={}
                    #for each base estimator a pipeline is created with name estimator_prefix+index (e.g. base_estimator1, base estimator2, ... if estimator_prefix=base estimator)
                    for index,estim in enumerate(estimator_list):
                        if normalizer: out={**out, f'{estimator_prefix}{index}__normalizer':hp.choice( f'pipe{index}_{dim_index}_{blender_index}__normalizer',[StandardScaler()])}
                        if isinstance(estim, type(PLSRegression())):
                            out={**out,f'{estimator_prefix}{index}__{dim_prefix}': hp.choice( f'reduce_{index}_{dim_index}_{blender_index}_dim_pls',['passthrough'])  }
                        else:
                            out={**out,**{f'{estimator_prefix}{index}__{key}': value for key, value in dim_red.items()}}
                        out={**out,**{f'{estimator_prefix}{index}__{key}': value for key, value in estim.items()}  }
                    #normalize the output of the base estimators
                    #prefix final_estimator is fixed in scikit stacking models (is possibly removed later when an estimator is used as blender instead of scikit stacking model)
                    if top_normalizer: 
                        out={**out, f'final_estimator__normalizer':hp.choice( f'final_estimator_{blender_index}_{dim_index}__normalizer',[StandardScaler()])}
                    else:  
                        out={**out, f'final_estimator__normalizer':hp.choice( f'final_estimator_{blender_index}_{dim_index}__normalizer',['passthrough'])}
                    out={**out,**{f'final_estimator__{key}': value for key, value in blender.items()}  }
                    parms.append(out)
        else:
            out={}
            final_prefix="final_estimator__"
            if top_method:
                final_prefix=""
            if top_normalizer: 
                out={**out, f'{final_prefix}normalizer':hp.choice( f'{final_prefix}{blender_index}__normalizer',[StandardScaler()])}
            else:  
                out={**out, f'{final_prefix}normalizer':hp.choice( f'{final_prefix}{blender_index}__normalizer',['passthrough'])}
            out={**out,**{f'{final_prefix}{key}': value for key, value in blender.items()}  }
            parms.append(out)
    return parms
"""implementation of the different stacking models.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""

import torch
import sys,  pandas as pd
import matplotlib.pyplot as plt
import numpy as np, math
from torch import nn

from automol.stacking_methodarchive import ClassifierArchive, ReducedimArchive, RegressorArchive 
from .stat_util import plot_reg_model
from .model_search import NestedCVModelSearch, NestedCVSingleModelSearch, NestedCVBaseStackingSearch, NestedCVSingleStackSearch, ClassificationFinder, RegressionFinder
from .grid_parameters import make_grid_parm, make_hyperopt_grid_parm, make_stacking_grid_parm, make_hyperopt_stacking_grid_parm
from .feature_generators import BottleneckTransformer, ECFPGenerator, retrieve_default_offline_generators
from .clustering import ClusteringAlgorithm, MurckoScaffoldClustering, ButinaSplitReassigned, HierarchicalButina, KmeansForSmiles
from .data_formation import SingleLigand, PairedLigands, get_nb_feature_multiplier


from .version import __version__ as AutoMLv
from packaging import version

from sklearn.ensemble import StackingRegressor,StackingClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report            
from sklearn.ensemble import  VotingClassifier            
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone

import time
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import  DataStructs

import subprocess
import socket
import sys
import json

from datetime import datetime
from typing import Callable



class MethodConfigurations:
    """    
    A class to retrieve the correct stacking model and corresponding parameters based on the given config string

    List of methods:
    :method __init__: constructor
    :method get_stacking_model: returns the stacking model given the Bottleneck model
    :method get_cv_params: returns all the parameter options in the required format using the given method archives
    """
    
    def __init__(self,classifier,distribution_defaults,hyperopt_defaults,model_config,regressionclassifier=False,n_jobs_dict=None):
        """    
        Initialization
        
        Args:
             classifier: boolean to indicate classification or regression
             distribution_defaults: boolean to indicate using distributional parameters
             hyperopt_defaults: boolean to indicate using hyperopt initialization
             model_config: string to represent choice of models: choose from [single_method,inner_methods, inner_stacking, single_stack, top_method, top_stacking, stacking_stacking]
             regressionclassifier:boolean to indicate use of regression methods for classification
             n_jobs_dict: dictionary containing the number of jobs [{'outer_jobs':None,'inner_jobs':-1,'method_jobs':2}]
        """
        self.classifier=classifier
        self.distribution_defaults=distribution_defaults
        self.hyperopt_defaults=hyperopt_defaults
        self.regressionclassifier=regressionclassifier
        if n_jobs_dict is None:
            self.n_jobs_dict={'outer_jobs':None,
                        'inner_jobs':-1,
                        'method_jobs':2}
        else:
            self.n_jobs_dict=n_jobs_dict
        self.inner_stacking_models=False
        self.single_stack=False
        self.top_stack=False
        self.top_method=False
        self.stacking_stackingmodels=False
        self.single_method=False
                
        if model_config=='inner_stacking':
            self.inner_stacking_models=True
        elif model_config=='single_stack':
            self.single_stack=True
        elif model_config=='top_method':
            self.top_method=True    
        elif model_config=='top_stacking':
            self.top_stack=True        
        elif model_config=='single_method':
            self.single_method=True
        elif model_config=='stacking_stacking':
            self.stacking_stackingmodels=True  
    
    def get_stacking_model(self,model,use_gpu=False,labelnames=None,compute_SD=True,feature_generators=None,
                           verbose=False, relative_modelling=False,
                            feature_operation:str='concat',property_operation:str='minus'):
        """    
        Returns the corresponding 'stacking model', e.g. the corresponding derived class of base class FeatureGenerationRegressor 
        
        Args:
             model: The Bottleneck model (encoder) for generating the features from SMILES [ChEMBL]
             use_gpu: boolean to indicate usage of gpu
             labelnames: the names of the classes for classification
             compute_SD: compute standard deviation on the output if possible
             feature_generators: dictionary of feature generators
             verbose:
        
        Returns: 
            the corresponding stacking model provided all the options
        """
        #storing feature generators for grid parameter generation
        self.feature_generators= feature_generators
        self.relative_modelling=relative_modelling
        
        if self.classifier:
            if self.inner_stacking_models:
                stacked_model=FeatureGenerationStackingClassifiers(model=model,use_gpu=use_gpu,compute_SD=compute_SD, feature_generators=feature_generators, labelnames=labelnames, outer_jobs=self.n_jobs_dict['outer_jobs'],verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation)
            elif self.single_stack:
                stacked_model=FeatureGenerationSingleStackClassifier(model=model,use_gpu=use_gpu,compute_SD=compute_SD, feature_generators=feature_generators, labelnames=labelnames, outer_jobs=self.n_jobs_dict['outer_jobs'],verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation)
            elif self.top_stack:
                stacked_model=FeatureGenerationTopstackingClassifier(model=model,use_gpu=use_gpu,compute_SD=compute_SD, feature_generators=feature_generators, labelnames=labelnames, outer_jobs=self.n_jobs_dict['outer_jobs'],verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation)
            elif self.single_method:
                stacked_model=FeatureGenerationSingleModelClassifier(model=model,use_gpu=use_gpu,compute_SD=compute_SD, feature_generators=feature_generators, labelnames=labelnames, outer_jobs=self.n_jobs_dict['outer_jobs'],verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation)
            elif self.stacking_stackingmodels:
                stacked_model=FeatureGenerationTopstackingStackingclassifier(model=model,use_gpu=use_gpu,compute_SD=compute_SD, feature_generators=feature_generators, labelnames=labelnames, outer_jobs=self.n_jobs_dict['outer_jobs'],verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation)
            else:
                stacked_model=FeatureGenerationClassifier(model=model,use_gpu=use_gpu,compute_SD=compute_SD, feature_generators=feature_generators, labelnames=labelnames, outer_jobs=self.n_jobs_dict['outer_jobs'],verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation)
        
        elif self.regressionclassifier:
            if self.inner_stacking_models:
                stacked_model=FeatureGenerationStackingRegressorsClassifier(model=model,use_gpu=use_gpu,compute_SD=compute_SD, feature_generators=feature_generators,  labelnames=labelnames, outer_jobs=self.n_jobs_dict['outer_jobs'],verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation)
            elif self.single_stack:
                stacked_model=FeatureGenerationSingleStackRegressorClassifier(model=model,use_gpu=use_gpu,compute_SD=compute_SD, feature_generators=feature_generators, labelnames=labelnames, outer_jobs=self.n_jobs_dict['outer_jobs'],verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation)
            elif self.top_stack:
                stacked_model=FeatureGenerationTopStackingRegressorClassifier(model=model,use_gpu=use_gpu,compute_SD=compute_SD, feature_generators=feature_generators, labelnames=labelnames, outer_jobs=self.n_jobs_dict['outer_jobs'],verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation)
            elif self.single_method:
                stacked_model=FeatureGenerationSingleModelRegressorClassifier(model=model,use_gpu=use_gpu,compute_SD=compute_SD, feature_generators=feature_generators, labelnames=labelnames, outer_jobs=self.n_jobs_dict['outer_jobs'],verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation)
            else:
                stacked_model=FeatureGenerationRegressionClassifier(model=model, use_gpu=use_gpu, compute_SD=compute_SD, feature_generators=feature_generators, labelnames=labelnames, outer_jobs=self.n_jobs_dict['outer_jobs'],verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation)
                           
        else:
            if self.inner_stacking_models:
                stacked_model=FeatureGenerationStackingRegressors( model=model,use_gpu=use_gpu,compute_SD=compute_SD, feature_generators=feature_generators, outer_jobs=self.n_jobs_dict['outer_jobs'] ,verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation )
            elif self.single_stack:
                stacked_model=FeatureGenerationSingleStackRegressor( model=model,use_gpu=use_gpu,compute_SD=compute_SD,  feature_generators=feature_generators, outer_jobs=self.n_jobs_dict['outer_jobs'],verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation )
            elif self.top_stack:
                stacked_model=FeatureGenerationTopStackingRegressor( model=model,use_gpu=use_gpu,compute_SD=compute_SD, feature_generators=feature_generators, outer_jobs=self.n_jobs_dict['outer_jobs'],verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation )
            elif self.single_method:
                stacked_model=FeatureGenerationSingleModelRegressor( model=model,use_gpu=use_gpu,compute_SD=compute_SD, feature_generators=feature_generators, outer_jobs=self.n_jobs_dict['outer_jobs'] ,verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation)
            else:
                stacked_model=FeatureGenerationRegressor( model=model,use_gpu=use_gpu,compute_SD=compute_SD, feature_generators=feature_generators, outer_jobs=self.n_jobs_dict['outer_jobs'],verbose=verbose, relative_modelling=relative_modelling, feature_operation=feature_operation,property_operation=property_operation  )
        return stacked_model
    
    def get_cv_params(self,method_list,dim_list,blender_list, method_archive=None,dim_archive=None,blender_archive=None, prefixes=None,normalizer=True, top_normalizer=False, random_state=42,local_dim_red=False, used_features=None):
        """    
        returns the prefixes with parameter options used in the different method parameters for base estimators, dimension reduction techniques and blender estimators.
        
        None Archive defaults consist of the hardcoded defaults for methods and their parameters using broad search space. 

        Args:
             method_list: list of strings corresponding to the method keys in method_archive 
             dim_list: list of strings corresponding to the method keys in dim_archive 
             blender_list: list of strings corresponding to the method keys in blender_archive 
             method_archive:  A derived class of method_archive holding the different methods used as base estimators (default:None)
             dim_archive: A derived class of method_archive holding the different methods used as dimensionality reduction techniques (default:None)
             blender_archive: A derived class of method_archive holding the different methods used as top estimators (default:None)
             prefixes: dictionary holding the different prefixes used in the pipelines for creatign the correct parameter options list (of lists) (default:None, e.g classifier: {'method_prefix':'clf','dim_prefix':'reduce_dim','estimator_prefix':'est_pipe'} and regressor: {'method_prefix':'reg','dim_prefix':'reduce_dim','estimator_prefix':'est_pipe'})
             normalizer: boolean to use normalizer for the base estimators (default:True)
             top_normalizer: boolean to use normalizer for the top estimator (default:False)
             random_state: random_state variable, use None if the experiments are not required to be reproducable (default:42)
             local_dim_red: boolean to indicate featurewise dimensionality reduction
             used_features: list of used features
        
        Returns: 
            tuple with the prefix dictionary, base estimator grid and final estimator grid
        """
        #defaults
        if prefixes is None:
            if self.classifier:
                prefixes={'method_prefix':'clf',
                           'dim_prefix':'reduce_dim',
                            'estimator_prefix':'est_pipe'}
            else: 
                prefixes={'method_prefix':'reg',
                           'dim_prefix':'reduce_dim',
                            'estimator_prefix':'est_pipe'}
        #defaults        
        if method_archive is None:
            if self.classifier:
                method_archive=ClassifierArchive(method_prefix=prefixes['method_prefix'], distribution_defaults=self.distribution_defaults, hyperopt_defaults=self.hyperopt_defaults, random_state=random_state)
            else:
                method_archive=RegressorArchive(method_prefix=prefixes['method_prefix'], distribution_defaults=self.distribution_defaults, hyperopt_defaults=self.hyperopt_defaults, random_state=random_state)
        if dim_archive is None:
                dim_archive=ReducedimArchive(method_prefix=prefixes['dim_prefix'], distribution_defaults=self.distribution_defaults, hyperopt_defaults=self.hyperopt_defaults, random_state=random_state) 
        if blender_archive is None:
                blender_archive=method_archive
                
        blender_params=None
        
        feature_splits=None
        nb_feat_factor=get_nb_feature_multiplier(relative=self.relative_modelling)
        if self.feature_generators is not None and local_dim_red:
            feature_splits={}
            col_splits=[0]
            if used_features is None:
                used_features=[key for key in self.feature_generators.keys()]
            for key in used_features:
                if key.startswith('fps') and key not in self.feature_generators.keys():
                    splits=key.split('_')
                    self.feature_generators[key]=ECFPGenerator(radius=splits[2], nBits =splits[1])
                assert key in self.feature_generators.keys(), f'provided features {key} is not in feature generator dictionary f{list(self.feature_generators.keys())}'
                col_splits.append(col_splits[-1]+(nb_feat_factor*self.feature_generators[key].get_nb_features()))
            feature_splits['col_splits']= col_splits
            feature_splits['names']= used_features
        
        if feature_splits is not None:
            if isinstance(dim_list[0],list):
                red_dim_list=[dim_archive.get_methods(dim_list_Feat) for dim_list_Feat in dim_list]
            else:
                red_dim_list=[ dim_archive.get_methods(dim_list) for f in range(len(used_features)) ]
        else:
            red_dim_list=dim_archive.get_methods(dim_list)
        
        method_list=method_archive.get_methods(method_list)
        blender_list=blender_archive.get_methods(blender_list)

        if self.hyperopt_defaults:
            #generate base estimators parameters
            if self.inner_stacking_models or self.stacking_stackingmodels:
                params_grid=make_hyperopt_stacking_grid_parm(normalizer=normalizer, blender_list=blender_list, estimator_list=method_list, dim_red_list=red_dim_list, dim_prefix=prefixes['dim_prefix'], estimator_prefix=prefixes['estimator_prefix'], top_normalizer=top_normalizer,feature_splits=feature_splits)
            else:
                params_grid=make_hyperopt_grid_parm(normalizer=normalizer,estimator_list=method_list,dim_red_list=red_dim_list,dim_prefix=prefixes['dim_prefix'],feature_splits=feature_splits)

            #generate top estimator parameters
            if self.single_stack:
                blender_params=make_hyperopt_stacking_grid_parm(normalizer=normalizer,blender_list=blender_list, estimator_list=method_list, dim_red_list=red_dim_list, dim_prefix=prefixes['dim_prefix'],estimator_prefix=prefixes['estimator_prefix'], top_normalizer=top_normalizer,feature_splits=feature_splits)
            elif self.single_method:
                if len(method_list)>1: print(f'Warning, more than one method given, using the first, e.g. {method_list[0]}')
                blender_params=[params_grid[0]]
            elif self.top_stack or self.top_method or self.stacking_stackingmodels:
                blender_params=make_hyperopt_stacking_grid_parm(normalizer=normalizer, blender_list=blender_list, estimator_list=None, dim_red_list=None, dim_prefix=prefixes['dim_prefix'], estimator_prefix=prefixes['estimator_prefix'], top_normalizer=top_normalizer,top_method=self.top_method,feature_splits=feature_splits)
        else:
            #generate base estimators parameters
            if self.inner_stacking_models or self.stacking_stackingmodels:
                params_grid=make_stacking_grid_parm(normalizer=normalizer, blender_list=blender_list, estimator_list=method_list, dim_red_list=red_dim_list, dim_prefix=prefixes['dim_prefix'], estimator_prefix=prefixes['estimator_prefix'], top_normalizer=top_normalizer,feature_splits=feature_splits)
            else:
                params_grid=make_grid_parm(normalizer=normalizer,estimator_list=method_list,dim_red_list=red_dim_list, dim_prefix=prefixes['dim_prefix'],feature_splits=feature_splits)

            #generate top estimator parameters
            if self.single_stack:
                blender_params=make_stacking_grid_parm(normalizer=normalizer,blender_list=blender_list, estimator_list=method_list, dim_red_list=red_dim_list, dim_prefix=prefixes['dim_prefix'],estimator_prefix=prefixes['estimator_prefix'], top_normalizer=top_normalizer,feature_splits=feature_splits)
            elif self.single_method:
                if len(method_list)>1: print(f'Warning, more than one method given, using the first, e.g. {method_list[0]}')
                blender_params=[params_grid[0]]
            elif self.top_stack or self.top_method or self.stacking_stackingmodels:
                blender_params=make_stacking_grid_parm(normalizer=normalizer, blender_list=blender_list, estimator_list=None, dim_red_list=None, dim_prefix=prefixes['dim_prefix'], estimator_prefix=prefixes['estimator_prefix'], top_normalizer=top_normalizer,top_method=self.top_method,feature_splits=feature_splits)
        
        return prefixes,params_grid,blender_params
        



    
############################################################
def plot_REG_model(model, X_test,  y_true,title=None, extend_axis_lim=0,  ax=None):
    """    
    plots scatter plot
    
    Args:
         model: model with predict functionality
         X_test: data matrix
         y_true: true values
         title: title for scatter plot
         extend_axis_lim: extend axis [=0]
         ax: matplotlib ax
    """
    y_pred= model.predict(X_test)
    metrics="MSE = %.2f , R2 = %.2f " % (mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred))
    line_X = np.arange(min(y_true.min(), y_pred.min())-extend_axis_lim, max(y_true.max(), y_pred.max()) +extend_axis_lim)
    lw = 2
    if ax:
        ax.scatter(y_true, y_pred, color='red', marker='.', label='Inliers')
        ax.plot(line_X, line_X, color='blue', linewidth=lw, label='regressor')
        #plt.legend(loc='lower right')
        ax.set_xlabel("y_true")
        ax.set_ylabel("y_pred")
        ax.set_title(title+ f':{metrics}')
        #ax.text(0.6,0.25, metrics)
        #ax.show()
    else:
        plt.scatter(y_true, y_pred, color='red', marker='.', label='Inliers')
        plt.plot(line_X, line_X, color='blue', linewidth=lw, label='regressor')
        #plt.legend(loc='lower right')
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        plt.title(title+ f':{metrics}')
        #plt.text(0.6,0.25, metrics)
        plt.show()



def compute_tanimoto_distances(smi_i,smiles_data,radius=2,nbits=1024):
    """    
    compute tanimoto distances (1- similarity) from one smiles to a list of smiles, creates morgan fps for all given smiles
    
    Args:
         smi_i: one smiles
         smiles_data: list of smiles
         radius: radius for fps [=2]
         nbits: nbits for fps [=1024]
    """
    #Fingerprints
    mol_i =Chem.MolFromSmiles(smi_i)
    fps_i= AllChem.GetMorganFingerprintAsBitVect(mol_i, radius, nbits)
    mols =[Chem.MolFromSmiles(s) for s in smiles_data]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]
    sims = DataStructs.BulkTanimotoSimilarity(fps_i, fps)
    return [1 - x for x in sims]




#################################################################
class BottleneckFeatureGenerator(nn.Module):
    """    
    Default features are the Bottleneck features generated from the bottleneckTransformer
    """
    def __init__(self,model='CHEMBL',use_gpu=False,batch_size=100,seq_len=220,multiprop_prefix='automol_multiprop', multiprop_split='@%@'):
        """    
        Initialization of the base feature generator
        
        Args:
             model: which encoder to be used in the bottleneck transformer
             use_gpu: boolean
             batch_size: batch size
             seq_len: max sequence length of the smiles
        """
        super(BottleneckFeatureGenerator, self).__init__()
        self.model=model
        self.use_gpu=use_gpu
        self.batch_size=batch_size
        self.seq_len=seq_len
        self.training_version=None
        self.multiprop_prefix=multiprop_prefix
        self.multiprop_split=multiprop_split
        
    
    def get_features(self, smiles,batch_size=100,seq_len=220):
        """    
        returns the bottleneck features
        
        Args:
             smiles: list of smiles
             batch_size: batch size
             seq_len: max sequence length of the smiles
        
        Returns: 
            numpy matrix with the features
        """
        if not hasattr(self,'bottleneck'): 
            self.bottleneck=BottleneckTransformer(model='CHEMBL')
                
        return self.bottleneck.generate(smiles)
    
    def is_FeatureGenerationRegressor(self):
        return False
    
    def is_FeatureGenerationClassifier(self):
        return False

    def is_FeatureGenerationRegressionClassifier(self):
        return False
    
    def generate_multi_property_name(self,property_list):
        return self.multiprop_prefix+self.multiprop_split+self.multiprop_split.join(property_list)

    def split_multi_property_name(self,name):
        return name.split(self.multiprop_split)[1:]

    def __call__(self, smiles,batch_size=100,seq_len=220):
        """    
        call self.get_features(smiles,batch_size,seq_len)
        """
        return self.generate_features( smiles,batch_size,seq_len)
################################################################
    
    
class FeatureGenerationRegressor(BottleneckFeatureGenerator):
    '''
    convert a Bottleneck model into a shallow regressor by using the features generator and other features as input for shallow regressor models
    
    provide a model class or a model pt file
    encoder_features (False): add encoder features
    tasks: can be 1) None (default)
                  2) list of tasks for which features are generated
                  3) 'all': all tasks in the model will be used
    '''
    def __init__(self,model=None, use_gpu=False,compute_SD=True,feature_generators=None, outer_jobs=None,verbose=False,relative_modelling=False,
                            feature_operation:str='concat',property_operation:str='minus'):
        '''
        Initialization
        
        Args:
             model: string indicating encoder
             use_gpu: boolean
             compute_SD: boolean to compute standard deviation of output
             feature_generators: dictionary with feature_generators [=None]
             outer_jobs: number of threads for outer cross-validation
             verbose
        '''
        ## training dataframe
        self.Train=None
        ## Validation dataframe
        self.Validation= None
        ## smiles column
        self.smiles= 'smiles'
        ## dictionary with the generated features
        self.FEATURES={'Train': {}, 'Validation':{}}
        BottleneckFeatureGenerator.__init__(self,model=model, use_gpu=use_gpu)
        ## dictionary containing the models with properties as keys
        self.models={}
        ## dictionary contain the base estimator per property
        self.base_estimators={}
        ## metrics
        self.metrics={}
        ## dictionary containing used features per property
        self.tasksfeatures_parameters={}
        ## method prefix dictionary
        self.prefix_dict={}
        ## groups of the training data set
        self.groups=None
        self.original_indices={'Train': None, 'Validation':None}
        self.pair_indices={'Train': None, 'Validation':None}
        ## boolean
        self.compute_SD=compute_SD
        if feature_generators is None:
            if not hasattr(self,'bottleneck'):
                self.bottleneck=BottleneckTransformer(model='CHEMBL')
                self.feature_generators=retrieve_default_offline_generators(model='CHEMBL', radius=2, nbits=2048)

        else:
            self.feature_generators=feature_generators
            if 'Bottleneck' not in feature_generators:
                if not hasattr(self,'bottleneck'): 
                    self.bottleneck=bottleneck=BottleneckTransformer(model='CHEMBL')
                self.feature_generators['Bottleneck']=self.bottleneck
        self.outer_jobs=outer_jobs
        self.verbose=verbose
        self.merged_model=False
        self.transformations_dict={}
        if relative_modelling:
            self.data_form=PairedLigands(feature_operation=feature_operation, property_operation=property_operation)
        else:
            self.data_form=SingleLigand()
        
        ## generated output when saving the model for a few public smiles
        self.reproducable_output=None
        ## public smiles
        self.reproducability_smiles=['CC1COC(=O)O1', 'N[C@@H](CO)C(O)=O', 'OCCNCCO', 
                                    'O=Cc1ccccc1',  'OCc1cccnc1', 'Nc1ccncc1N', 
                                    'C\C=C\C=C\C(O)=O', 'CC1CCCC(C)N1', 'CCCCCC(C)N', 
                                    'C[C@@H](O)[C@H](N)C(O)=O','OC(=O)c1ccccc1', 'OC(=O)c1ccccc1', 
                                    'Cc1cc(C)c(=O)[nH]n1', 'CC1OC(C)OC(C)O1', 
                                    'OC(=O)C1CSCN1', 'Cl[Al](Cl)Cl', 'O=S1(=O)CCNCN1', 
                                    'Oc1ccc2ccccc2c1',   'CC(C)N1CCNC1=S',  'NC(CCC(O)=O)C(O)=O',
                                     'CC(N)Cc1ccccc1C',  'OC(C(O)C(O)=O)C(O)=O', 
                                    'CC1=CC(=O)[C@H]2C[C@@H]1C2(C)C',  'CCC1(C)CC(=O)NC(=O)C1', 'CC(C)C1CCC(C)CC1O', 
                                    'CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O','CCCC1OC1(CC)C(N)=O', 'COCC(O)COC(C)(C)C', 
                                    'CCNC(C)Cc1ccccc1', 'OC(=O)CCSCC(O)=O', 
                                    'OC(=O)CCCc1ccccc1',  'CCCc1ncc(s1)C(O)=O', 
                                    'OCC(CO)OP(O)(O)=O',  'OC1C(=O)C(=O)C(O)C(=O)C1=O', 
                                    'OCC(CO)(CCl)CCl', 'CC(=O)N1CC(O)C[C@H]1C(O)=O',  'Oc1ccccc1S(O)(=O)=O', 
                                    'N[C@@H](CCCNC(N)=N)C(O)=O',  'OC[C@H](O)[C@H]1OC(=O)C(O)=C1O', 
                                    'OC[C@H](O)[C@H]1OC(=O)C(O)=C1O',  'OC(C1CCC=CC1)P(O)=O', 
                                    'CN1[C@@H](CCC1=O)c1cccnc1', 'OC(=O)C1CSC(N1)C(O)=O', 
                                    'OC[C@H]1OC(=O)[C@H](O)[C@@H](O)[C@@H]1O']

    def generate_reproducable_output(self):
        """    
        Generate and save output for fixed set of SMILES to be compared on different machine for reproducable output. Function is required to be executed before saving the model.
        
        Returns: 
            dictionary with the maximum error per failed prediction key
        """
        assert self.models, 'Train model(s) first'
        old_sd=self.compute_SD
        self.compute_SD=True
        indices=[]
        n=len(self.reproducability_smiles)
        for i in range(n):
            for j in range(i+1,n):
                indices.append((i,j))
        self.reproducable_output=self.predict( props =None, smiles=self.reproducability_smiles,convert_log10=False, original_indices=indices, indices=indices)
        self.compute_SD=old_sd
    def test_reproducable_output(self,rtol=1e-05, atol=1e-08,verbose=False,relative_error=False, numeric_zero=1e-15):
        old_sd=self.compute_SD
        self.compute_SD=True
        max_err_failed_props={}
        if self.reproducable_output:
            
            indices=[]
            n=len(self.reproducability_smiles)
            for i in range(n):
                for j in range(i+1,n):
                    indices.append((i,j))
            new_output=self.predict( props =None, smiles=self.reproducability_smiles,convert_log10=False, original_indices=indices, indices=indices)
            self.compute_SD=old_sd
            for pp in [k for k in new_output]:
                if not pp.startswith('predicted_labels'):
                    old_p=self.reproducable_output[pp].astype(float)
                    new_p=new_output[pp].astype(float)
                    if not np.ma.allclose(new_p,old_p,rtol=rtol,atol=atol):
                        if relative_error:
                            max_err_failed_props[f'{pp}_max_relative_error']=np.amax(np.absolute(new_p-old_p/old_p, where=(np.absolute(old_p)>numeric_zero)))
                        max_err_failed_props[f'{pp}_max_absolute_error']=np.amax(np.absolute(new_p-old_p))
                        if verbose: 
                            max_val=max_err_failed_props[f'{pp}_max_absolute_error']
                            print(f'allclose failed for property {pp} with maximum absolute error of {max_val}')
            if not max_err_failed_props and verbose: print('Success! All values of reproduced output are close!')
        else:
            print('No reproducable output generated yet, call generate_reproducable_output before saving model or use the static method save_model')
        return max_err_failed_props
                
    ##########################################
    def search_model(self,df, # df with smiles and the properties
                    prop,# name of the property in df
                    blender_props=None,
                    smiles=None,
                    params_grid=None, # dict for grid search
                    paramsearch=None,
                    include_rdkitfeatures=False, ## add redkit features
                    include_fps = False , radius=2, nBits =2048,## add fps
                    include_Bottleneck_features=True, ## include encoder in features
                     features=None,
                    blender_properties=[],
                    scoring='r2',
                    cv=3, ## inner loop for CV
                    outer_cv_fold=4, ## outer loop for CV
                    n_jobs=-1,
                    use_memory=False,
                   plot_validation=False,
                   refit=False,# refit with entire data=train +validation after validation
                    split='SKF' # CV split : SKF , GKF, LGO
                    ,n_iter=None #number of iterations for randomized search, if None then grid search is performed
                     ,blender_params=None
                     ,prefix_dict=None
                     ,random_state=42
                     ,sample_weight=None
                     ,results_dict=None
                   ):
        """    
        Finds a model for one property using the given estimator list and parameters
        
        Args:
             df: the pandas data frame
             prop: the property/column of df with the target values
             smiles: the smiles column
             params_grid: the base estimator parameter grid
             include_rdkitfeatures:[deprecated] boolean to include rdkit descriptors (default:False)
             include_fps:[deprecated] boolean to include morgan fingerprints (default:False)
             include_Bottleneck_features:[deprecated] boolean to include Bottleneck transformer features (default:True)
             scoring: string representing scoring function used in scikit-learn (default:'r2')
             cv: number of folds for inner cross-validation (default:3)
             outer_cv_fold: number of folds for outer cross-validation (default:4)
             n_jobs: number of threads used (default:-1)
             use_memory: use memory in pipeline (default:False)
             plot_validation: boolean to plot validation results (default:False)
             refit: refit models using validation data (default:True)
             split: string to select options from different splits for cross-validation (default:SKF)
             n_iter: number of iterations randomized_search, if None grid_search is performed(default:None)
             blender_params: the top estimator parameter grid
             prefix_dict: dictionary of prefix strings used in the paramter grids
             random_state: random state initialization value
             features: list of used feature keys
             blender_properties: list of additional properties in the dataframe given directly to the blender
        """
        print('#################################################################')
        print(f"Training a model for {prop}")
        if not smiles: smiles=self.smiles
        
        if isinstance(prop,list):
            assert isinstance(df, type(None))  and hasattr(self, 'Train')
            for p in prop:
                assert p in self.Train.columns
            merge_name=self.generate_multi_property_name(prop)
            self.Train[merge_name]=list(np.concatenate([np.array([self.Train[p].values]).T for p in prop],axis=1))
            if self.Validation is not None:
                self.Validation[merge_name]=list(np.concatenate([np.array([self.Validation[p].values]).T for p in prop],axis=1))
            feature_na=np.zeros(len(self.Train[merge_name]), dtype=bool)
            for i, p in enumerate(prop):
                feature_na=np.logical_or(self.Train[p].isna(),feature_na)
            self.Train.loc[feature_na,merge_name] =np.nan  
            prop=merge_name
        
        #storing package version
        if self.training_version is None: self.training_version=AutoMLv
                
        #print('TASK',prop, add_features_tasks , isinstance(add_features_tasks, str))
        if features is None:
            features =[]
            if include_Bottleneck_features:
                features.append('Bottleneck')
                compute_Bottleneck=True
            else :compute_Bottleneck= False
            if include_rdkitfeatures:features.append('rdkit')
            if include_fps:
                features.append(f'fps_{nBits}_{radius}')
                if not f'fps_{nBits}_{radius}'in self.feature_generators:
                    features.append(f'fps_{nBits}_{radius}')
                    self.feature_generators[f'fps_{nBits}_{radius}']=ECFPGenerator(radius=radius, nBits =nBits)
        else:
            for feat_key in features:
                if feat_key not in self.feature_generators:
                    if feat_key.startswith('fps'):
                        splits=feat_key.split('_')
                        self.feature_generators[feat_key]=ECFPGenerator(radius=splits[2], nBits =splits[1])
                assert feat_key in self.feature_generators, 'provided feature key not available in provided feature generators'
                if self.feature_generators[feat_key].get_nb_features()>5000:
                    print(f'\033[1m Warning \033[0m: Number of features for generator {feat_key} is larger than 5000, training will consume a lot of resources/time')
                
        self.tasksfeatures_parameters[prop]={'features':features,'blender_properties':blender_properties}
        if self.verbose:  print('using the following features', features)
        if isinstance(df, type(None))  and hasattr(self, 'Train'):
            assert prop in self.Train.columns
            
            self.original_indices['Train'],self.pair_indices['Train']=self.data_form.get_pairs(self.Train,pairs_col='pairs')
            na=self.Train[smiles].isna()
            if(sum(na)):
                if self.verbose:
                    print(f'Removing the following {sum(na)} nan smiles')
                    print(self.Train[~na,smiles])
                self.Train=self.Train[~na]
            
            if not self.FEATURES['Train']: self.FEATURES['Train']={}
            self.FEATURES['Train']=self.data_form.precompute_features(smiles=self.Train[smiles],gen_features=self.FEATURES['Train'],feature_generators=self.feature_generators,feature_list=features, original_indices=self.original_indices['Train'], indices=self.pair_indices['Train'])
            X_train,y_train,na= self.data_form.create_X_y(self.Train,prop, smiles,self.FEATURES['Train'],features, prop.startswith(self.multiprop_prefix),indices=self.pair_indices['Train'])
            blender_l=[]
            for p_blen in blender_properties:
                assert p_blen in self.Train.columns
                _,X_blender_i,na_i= self.data_form.create_X_y(self.Train,p_blen, smiles,self.FEATURES['Train'],features, False,indices=self.pair_indices['Train'])
                blender_l=[b[~na_i[~na]] for b in blender_l]
                blender_l.append(X_blender_i[~na[~na_i]])
                X_train=X_train[~na_i[~na]]
                y_train=y_train[~na_i[~na]]
                na=np.logical_or(na,na_i)
            if len(blender_l)>0:
                blender_l=[np.expand_dims(a, axis=1) if len(a.shape)<2 else a for a in blender_l] 
                X_blender=np.hstack(blender_l)
            else:
                X_blender=None

            #X_train,y_train,na= self._create_X_y(self.Train,prop, 'Train',smiles,self.FEATURES['Train'],features)
        else:
            assert prop in df.columns and smiles in df.columns
            na=df[prop].isna() | df[smiles].isna()
            SM=df.loc[~na, smiles]
            
            feats=self.data_form.precompute_features(smiles=SM,feature_generators=self.feature_generators,feature_list=features,indices=None)
            feature_na=np.zeros(len(SM), dtype=bool)
            for i, feature_name in enumerate(features):
                feature_na=np.logical_or([ np.isnan(row).any() for j, row in enumerate(feats[feature_name])],feature_na)
            if self.verbose:
                print(f'Deleted the following smiles from training due to nan features/property value:{list(SM[feature_na])}')
            y_train=df.loc[~np.logical_or(na,feature_na),prop].values
            X_train=np.concatenate( [feats[k][~feature_na] for k in features], axis=-1)
            X_blender=None

        if sample_weight is not None: sample_weight=sample_weight[~na]
        if self.verbose: print('X_train.shape ',X_train.shape ,'y_train.shape ', y_train.shape)
        try:
            self.groups=self.data_form.update_groups(self.groups,self.pair_indices['Train'])
            groups= self.groups[~na]
        except ValueError:
            print('Cluster the compounds by using the method Data_clustering!')
            
        if prefix_dict is not None: self.prefix_dict=prefix_dict
        
        start_kfold=time.time()
        self.do_kfold_search(prop,X=X_train, y=y_train,X_blender=X_blender, groups=groups,params_grid=params_grid,paramsearch=paramsearch, scoring=scoring ,cv=cv,
                             use_memory=use_memory, outer_cv_fold=outer_cv_fold,
                             split=split,blender_params=blender_params,prefix_dict=prefix_dict,random_state=random_state,sample_weight=sample_weight)
        hours, minutes_rem = divmod(time.time() - start_kfold, 60*60)
        minutes, seconds = divmod(minutes_rem, 60)
        if results_dict is not None:
            results_dict[prop]['execution time']=f'{int(hours)}h{int(minutes)}m{int(seconds)}s'
        if self.verbose: 
            print(f'performed complete estimator fit in {minutes} min and {seconds} seconds' )
       
        if  isinstance(self.Validation, pd.DataFrame) :
            if plot_validation:
                if self.verbose: print('Validation using the validation dataset')
                self.validate(df=None,  props=prop, true_props=None,smiles=smiles)
            if refit:
                if self.verbose: print('Refiting the model using the entire dataset')
                na=self.Validation[smiles].isna()
                if(sum(na)):
                    if self.verbose: 
                        print(f'  Removing the following {sum(na)}nan smiles')
                        print(self.Validation[~na,smiles])  
                    self.Validation=self.Validation[~na]
                if not self.FEATURES['Validation']: self.FEATURES['Validation']={}
                self.original_indices['Validation'],self.pair_indices['Validation']=self.data_form.get_pairs(self.Validation,pairs_col='pairs')
                self.FEATURES['Validation']=self.data_form.precompute_features(smiles=self.Validation[smiles],gen_features= self.FEATURES['Validation'], feature_generators=self.feature_generators,feature_list=features,original_indices=self.original_indices['Validation'], indices=self.pair_indices['Validation'])
                X_valid,y_valid,nav= self.data_form.create_X_y(self.Validation,prop, smiles,self.FEATURES['Validation'],features, prop.startswith(self.multiprop_prefix),indices=self.pair_indices['Validation'])
                #X_valid,y_valid,nav= self._create_X_y(self.Validation,prop, 'Validation',smiles,self.FEATURES['Validation'],features)
                blender_l=[]
                for p_blen in blender_properties:
                    assert p_blen in self.Train.columns
                    _,X_blender_i,na_i= self.data_form.create_X_y(self.Validation,p_blen, smiles,self.FEATURES['Validation'],features, False,indices=self.pair_indices['Validation'])
                    blender_l=[b[~na_i[~nav]] for b in blender_l]
                    blender_l.append(X_blender_i[~nav[~na_i]])
                    X_valid=X_valid[~na_i[~nav]]
                    y_valid=y_valid[~na_i[~nav]]
                    nav=np.logical_or(nav,na_i)
                
                if len(blender_l)>0:
                    blender_l=[np.expand_dims(a, axis=1) if len(a.shape)<2 else a for a in blender_l] 
                    X_blender_val=np.hstack(blender_l)
                else:
                    X_blender_val=None


                if X_blender is not None:
                    X_all=np.concatenate( [np.hstack((X_train,X_blender)),np.hstack((X_valid,X_blender_val))], axis=0)
                else:
                    X_all=np.concatenate( [X_train,X_valid], axis=0)
                y_all=np.concatenate( [y_train,y_valid], axis=0)
                if self.verbose > 1: print('X_all.shape',X_all.shape ,'y_all.shape', y_all.shape)
                if isinstance(self.models[prop], list): 
                    self.models[prop]= [clone(est).fit(X_all, y_all) for est in self.models[prop]]
                else:
                    self.models[prop]= clone(self.models[prop]).fit(X_all, y_all)
                if prop in self.base_estimators:
                    self.base_estimators[prop]= [clone(est).fit(X_all, y_all) for est in self.base_estimators[prop]]   
    
    def recompute_features(self,smiles=None):
        if not smiles: smiles=self.smiles
        allprops=[p for p in  self.models]
        self.FEATURES['Train']={}
        self.FEATURES['Validation']={}
        for p in  allprops:
            features =self.tasksfeatures_parameters[p]['features']

            self.original_indices['Train'],self.pair_indices['Train']=self.data_form.get_pairs(self.Train,pairs_col='pairs')
            na=self.Train[smiles].isna()
            if(sum(na)):
                if self.verbose:
                    print(f'Removing the following {sum(na)} nan smiles')
                    print(self.Train[~na,smiles])
                self.Train=self.Train[~na]
            self.FEATURES['Train']=self.data_form.precompute_features(smiles=self.Train[smiles],gen_features=self.FEATURES['Train'],feature_generators=self.feature_generators,feature_list=features, original_indices=self.original_indices['Train'], indices=self.pair_indices['Train'])
    
            na=self.Validation[smiles].isna()
            if(sum(na)):
                if self.verbose: 
                    print(f'  Removing the following {sum(na)}nan smiles')
                    print(self.Validation[~na,smiles])  
                self.Validation=self.Validation[~na]

            self.original_indices['Validation'],self.pair_indices['Validation']=self.data_form.get_pairs(self.Validation,pairs_col='pairs')
            self.FEATURES['Validation']=self.data_form.precompute_features(smiles=self.Validation[smiles],gen_features= self.FEATURES['Validation'], feature_generators=self.feature_generators,feature_list=features,original_indices=self.original_indices['Validation'], indices=self.pair_indices['Validation'])
            

    def create_modfinder(self):
        """    
        Returns the appropriate derivation of ModelFinder
        
        Returns: 
            RegressionFinder(verbose,NestedCVModelSearch,outer_jobs)
        """
        return RegressionFinder(self.verbose,NestedCVModelSearch(self.verbose),outer_jobs=self.outer_jobs)

    def do_kfold_search(self,prop,X, y,X_blender, groups,params_grid,paramsearch, scoring ,cv=3 ,use_memory=False, outer_cv_fold=5, split='GKF',blender_params=None,prefix_dict=None,random_state=42,sample_weight=None):
        """    
        performs the nested cross-validation for regression models and stores the found estimators
        
        If no top estimator, then the average of the predictions of the models are taken
        else the output is given of the top estimator
        
        Args:
             prop: the property/column of df with the target values
             X: the feature matrix
             y: the target values corresponding to the samples of X
             X_blender: matrix of property values/features directly fed into the blender
             groups: the non-overlapping groups
             params_grid: the base estimator parameter grid
             scoring: string representing scoring function used in scikit-learn (default:'r2')
             cv: number of folds for inner cross-validation (default:3)
             outer_cv_fold: number of folds for outer cross-validation (default:4)
             n_jobs: number of threads used (default:-1)
             use_memory: use memory in pipeline (default:False)
             split: string to select options from different splits for cross-validation (default:GKF)
             n_iter: number of iterations randomized_search, if None grid_search is performed(default:None)
             blender_params: the top estimator parameter grid
             prefix_dict: dictionary of prefix strings used in the paramter grids
             random_state: random state initialization value
             sample_weight: sample weights
        """
        modfinder=self.create_modfinder()
        CV= modfinder.model_search( X=X, y=y, X_blender=X_blender,groups=groups,params_grid=params_grid,paramsearch=paramsearch, scoring=scoring ,cv=cv ,
                                use_memory=False,verbose=self.verbose, outer_cv_fold=outer_cv_fold
                                ,split=split,blender_params=blender_params,prefix_dict=prefix_dict,random_state=random_state,sample_weight=sample_weight)
        if X_blender is not None:
            X=np.hstack((X,X_blender))
        if blender_params is not None and CV['blender_model'] is not None:
            if isinstance(CV['blender_model'],  type(StackingRegressor([None]))):
                self.models[prop]= clone(CV['blender_model']).fit(X,y)
                self.base_estimators[prop] =[clone(m[1]).fit(X,y) for m in self.models[prop].estimators]
            else:
                if sample_weight is not None:
                    fit_params={}
                    est_prefix=prefix_dict['method_prefix']
                    red_prefix=prefix_dict['dim_prefix']
                    fit_params[f'{red_prefix}__{est_prefix}__sample_weight']=sample_weight
                    fit_params[f'{est_prefix}__sample_weight']=sample_weight
                    self.models[prop]= clone(CV['blender_model']).fit(X,y,**fit_params)
                    del fit_params[f'{red_prefix}__{est_prefix}__sample_weight']
                    if hasattr(self.models[prop].steps[1][1], 'model_list'):
                        self.base_estimators[prop] =[clone(m).fit(X,y,**fit_params) for m in self.models[prop].steps[1][1].model_list]
                else:
                    self.models[prop]= clone(CV['blender_model']).fit(X,y)
                    if hasattr(self.models[prop].steps[1][1], 'model_list'):
                        self.base_estimators[prop] =[clone(m).fit(X,y) for m in self.models[prop].steps[1][1].model_list]
        else:
            if sample_weight is not None:
                fit_params={}
                est_prefix=prefix_dict['method_prefix']
                fit_params[f'{est_prefix}__sample_weight']=sample_weight
                self.models[prop]= [clone(m).fit(X,y,**fit_params) for m in CV['models']]
            else:
                self.models[prop]= [clone(m).fit(X,y) for m in CV['models']]
        #self.models[prop]= [m for m in CV['models']]
        self.metrics[prop]=CV['Nested_CV score']
        
    ############################
    def refit_model(self , models= None ,smiles=None,use_validation=True,sample_train=None,sample_val=None,prefix_dict=None):
        '''
        refit all the models to the entire datasets train + validation
        
        Args:
             models: list of properties/models to be refitted [=None], with None all available properties
             smiles: list of smiles
             use_validation: add validation data
             sample_train: sample weights for the training data
             sample_val: sample weights for the validation data
        '''
        if smiles is None: smiles=self.smiles
        assert self.FEATURES['Train'] or not self.FEATURES['Validation'], 'compute features first (run validation)'
        #self.precompute_features(smiles, data='Train')
        if models is None: models=self.models
        elif isinstance(models, (list, tuple)): models= models
        elif isinstance(models, str): models=[models]
        for m in models:
            if self.verbose:
                if use_validation:
                    print(f'refit the model {m} to the entire data (train+validation)')
                else:
                    print(f'refit the model {m} to the train data')
            if m.startswith(self.multiprop_prefix):
                multi_prop=self.split_multi_property_name(m)
                for p in multi_prop:                            
                    assert p in self.Train.columns, f'{p} not in Train set'
                    assert p in self.Validation.columns, f'{p} not in Validation set'
                    
                self.Train[m]=list(np.concatenate([np.array([self.Train[p].values]).T for p in multi_prop],axis=1))
                if self.Validation is not None:
                    self.Validation[m]=list(np.concatenate([np.array([self.Validation[p].values]).T for p in multi_prop],axis=1))
                    feature_nav=np.zeros(len(self.Validation[m]), dtype=bool)
                    for i, p in enumerate(multi_prop):
                        feature_nav=np.logical_or(self.Validation[p].isna(),feature_nav)
                    self.Validation.loc[feature_nav,m] =np.nan  
                feature_na=np.zeros(len(self.Train[m]), dtype=bool)
                for i, p in enumerate(multi_prop):
                    feature_na=np.logical_or(self.Train[p].isna(),feature_na)
                self.Train.loc[feature_na,m] =np.nan  
                
            
            assert m in self.models, f'{m} not in models'
            assert m in self.Train.columns, f'{m} not in Train set'
            assert m in self.Validation.columns, f'{m} not in Validation set'
            
            features =self.tasksfeatures_parameters[m]['features']
            #self.precompute_features(smiles, data='All', feature_list=features)
            if not self.FEATURES['Train']: self.FEATURES['Train']={}
            if not self.FEATURES['Validation']: self.FEATURES['Validation']={}
                
            self.original_indices['Train'],self.pair_indices['Train']=self.data_form.get_pairs(self.Train,pairs_col='pairs')           
            self.original_indices['Validation'],self.pair_indices['Validation']=self.data_form.get_pairs(self.Validation,pairs_col='pairs')
            self.FEATURES['Train']=self.data_form.precompute_features(smiles=self.Train[smiles],gen_features=self.FEATURES['Train'],feature_generators=self.feature_generators,feature_list=features,original_indices=self.original_indices['Train'], indices=self.pair_indices['Train'])
            self.FEATURES['Validation']=self.data_form.precompute_features(smiles=self.Validation[smiles],gen_features=self.FEATURES['Validation'],feature_generators=self.feature_generators,feature_list=features, original_indices=self.original_indices['Validation'], indices=self.pair_indices['Validation'])
            X_train,y_train,na= self.data_form.create_X_y(self.Train,m, smiles,self.FEATURES['Train'],features, m.startswith(self.multiprop_prefix),indices=self.pair_indices['Train'])
            #X_train,y_train,na= self._create_X_y(self.Train,m, 'Train',smiles,self.FEATURES['Train'],features)
            
            X_valid,y_valid,nav= self.data_form.create_X_y(self.Validation,m, smiles,self.FEATURES['Validation'],features, m.startswith(self.multiprop_prefix),indices=self.pair_indices['Validation'])
            #X_valid,y_valid,nav= self._create_X_y(self.Validation,m, 'Validation',smiles,self.FEATURES['Validation'],features)
            
            blender_properties =self.tasksfeatures_parameters[m]['blender_properties']
            blender_l=[]
            blender_l_val=[]
            for p_blen in blender_properties:
                assert p_blen in self.Train.columns

                _,X_blender_i,na_i= self.data_form.create_X_y(self.Train,p_blen, smiles,self.FEATURES['Train'],features, False,indices=self.pair_indices['Train'])
                blender_l=[b[~na_i[~na]] for b in blender_l]
                blender_l.append(X_blender_i[~na[~na_i]])
                X_train=X_train[~na_i[~na]]
                y_train=y_train[~na_i[~na]]
                na=np.logical_or(na,na_i)

                _,X_blender_i,na_i= self.data_form.create_X_y(self.Validation,p_blen, smiles,self.FEATURES['Validation'],features, False,indices=self.pair_indices['Validation'])
                blender_l_val=[b[~na_i[~nav]] for b in blender_l_val]
                blender_l_val.append(X_blender_i[~nav[~na_i]])
                X_valid=X_valid[~na_i[~nav]]
                y_valid=y_valid[~na_i[~nav]]
                nav=np.logical_or(nav,na_i)
            
            if len(blender_l)>0:
                blender_l=[np.expand_dims(a, axis=1) if len(a.shape)<2 else a for a in blender_l] 
                X_blender_val=np.hstack(blender_l)
                X_train=np.hstack((X_train,X_blender_val))

            if len(blender_l_val)>0:
                blender_l_val=[np.expand_dims(a, axis=1) if len(a.shape)<2 else a for a in blender_l_val] 
                X_blender_val=np.hstack(blender_l_val)
                X_valid=np.hstack((X_valid,X_blender_val))
            

            if use_validation:
                #fitted_on_validation used in computing errors in stacking_models (requires fits of the base estimators, see predict property)
                X_all=np.concatenate( [X_train,X_valid], axis=0)
                y_all=np.concatenate( [y_train,y_valid], axis=0)
            else:
                X_all=np.concatenate( [X_train], axis=0)
                y_all=np.concatenate( [y_train], axis=0)
                
            if self.verbose > 1: print('model ',m,'X_all.shape',X_all.shape ,'y_all.shape', y_all.shape)
            
            sample_weight=None
            if sample_train is not None:
                if use_validation and sample_val is not None:
                    sample_weight=np.concatenate( [sample_train[~na],sample_val[~nav]], axis=0)
                elif not use_validation:
                    sample_weight=sample_train[~na]
            if isinstance(self.models[m], list): 
                if sample_weight is not None :
                    fit_params={}
                    est_prefix=prefix_dict['method_prefix']
                    fit_params[f'{est_prefix}__sample_weight']=sample_weight
                    self.models[m]= [clone(est).fit(X_all, y_all,**fit_params) for est in self.models[m]]
                else:
                    self.models[m]= [clone(est).fit(X_all, y_all) for est in self.models[m]]
            else:
                if sample_weight is not None and isinstance(self.models[m],Pipeline):
                    fit_params={}
                    est_prefix=prefix_dict['method_prefix']
                    red_prefix=prefix_dict['dim_prefix']
                    fit_params[f'{red_prefix}__{est_prefix}__sample_weight']=sample_weight
                    fit_params[f'{est_prefix}__sample_weight']=sample_weight
                    self.models[m]= clone(self.models[m]).fit(X_all, y_all,**fit_params)
                else:
                    self.models[m]= clone(self.models[m]).fit(X_all, y_all)
            if m in self.base_estimators:
                if sample_weight is not None:
                    fit_params={}
                    est_prefix=prefix_dict['method_prefix']
                    fit_params[f'{est_prefix}__sample_weight']=sample_weight
                    self.base_estimators[m]= [clone(est).fit(X_all, y_all,**fit_params) for est in self.base_estimators[m]]
                else:
                    self.base_estimators[m]= [clone(est).fit(X_all, y_all) for est in self.base_estimators[m]]

    ##########################################
    def predict_empty_smiles(self,allprops,compute_SD=True,convert_log10=True):
        """    
        creates empty output but with the right format
        
        Args:
             allprops: the properties
             compute_SD: add standard deviation
             convert_log10: boolean to revert to original values
        Returns: 
            dictionary with all the keys but empty arrays
        """
        pred={}
        props=[]
        for p in allprops:
            if p.startswith(self.multiprop_prefix):
                multi_prop=self.split_multi_property_name(p)
                for mp in multi_prop:
                    props.append(mp)  
            else:
                props.append(p)
        for p in props:
            pred[f'predicted_{p}']=np.array([],dtype=object)
            if compute_SD:
                pred[f'predicted_{p}_std']=np.array([],dtype=object)
        if convert_log10:
            logk= [k for k in pred.keys() if k.startswith('predicted_log10') ]
            logitk= [k for k in pred.keys() if k.startswith('predicted_logit') ]
            if len(logk):
                for p in logk:
                    tp='_'.join(p.split('_')[2:])
                    pred[f'predicted_{tp}']=pred[p]
                    pred.pop(p)
            if len(logitk):
                for p in logitk:
                    tp='_'.join(p.split('_')[2:])
                    pred[f'predicted_{tp}']=pred[p]
                    pred.pop(p)
        return pred
    
    def add_predict_transformation_for_p(self,p:str,*,transformation: Callable[[np.ndarray], np.ndarray]=None):
        """
            adds an output transformation for a specific property prediction. The property is predicted and then transformed using the given function
            
            Args:
                p: the trained property name, the name shown in the logs
                transformation: a function that can be called using () (__call__ function implemented for classes)
        """
        assert p in self.models, f'Property {p} not trained'
        if transformation is not None:
            try:
                transformation(np.array([1]))
                self.transformations_dict[p]=transformation
            except:
                print('Calling transform failed, not transformation added, implement __call__(np.array) in the class' )
        pass
            
        
    
    def predict(self, smiles,props=None, blender_properties_dict={}, batch_size=50,seq_len=None ,
                    compute_SD='model_control', convert_log10=True, original_indices=None, indices=None):
        """    
        predicts the output for the given smiles and properties
        
        Args:
             smiles: list of smiles
             props: list of properties/models in the stacking model
             batch_size: [unused] added to match transformer interface
             seq_len: sequence length of the smiles for the tokenizer
             compute_SD: to indicate standard deviation computation, ['model_control', True, False], if model_control than the class variable is used
             convert_log10: revert log10 or logit predictions back to their original values [=TRUE]
        
        Returns: 
            dictionary of all the predictions for the different models
        """
        if hasattr(self,'training_version'):
            if version.parse(AutoMLv) < version.parse(self.training_version):
                print(f"Training AutoMoL version {self.training_version} is further ahead than the current AutoMoL version: {AutoMLv}")
                assert  version.parse(self.training_version) >= version.parse("0.4.2") and version.parse(AutoMLv) < version.parse("0.4.2"), 'Used AutoML version is incompatible with Training version, update to a more recent version'
        
        
        if compute_SD == 'model_control':
             compute_SD=self.compute_SD
        elif compute_SD not in [True ,False]:
            compute_SD=False
        if not props:
            allprops=[p for p in  self.models]
        elif isinstance(props, (list, tuple)) :
            allprops=[]
            for p in props:
                assert p in  self.models
                allprops.append(p)
        else:
            allprops=[props]
        if smiles is not None:
            if isinstance(smiles, (list, tuple, np.ndarray,pd.Series)):
                #if len(smiles)>0:
                #    smiles=[smi for smi in smiles if len(smi)>0 ]
                if len(smiles)<1:
                    return self.predict_empty_smiles(allprops,compute_SD,convert_log10)
            else:
                smiles=[smiles]
        #print('predicting the following props:',allprops)
        if not seq_len:
            seq_len= max([len(s) if s is not None else 0 for s in smiles]) +5
        seq_len= min(seq_len,220)
        
        #for older models
        if not hasattr(self,'feature_generators'):
            if not hasattr(self,'bottleneck'):   
                self.bottleneck=BottleneckTransformer(model='CHEMBL')
                self.feature_generators=retrieve_default_offline_generators(model='CHEMBL', radius=2, nbits=2048)
                if 'Bottleneck' not in self.feature_generators:
                    self.feature_generators['Bottleneck']=self.bottleneck
                                             
        feats={}
        for p in  allprops:
            features =self.tasksfeatures_parameters[p]['features']
            for feat_key in features:
                if feat_key not in self.feature_generators:
                    if feat_key.startswith('fps'):
                        splits=feat_key.split('_')
                        self.feature_generators[feat_key]=ECFPGenerator(radius=splits[2], nBits =splits[1])
                assert feat_key in self.feature_generators, 'provided feature generation key not available'
                
        #
        for p in  allprops:
            features =self.tasksfeatures_parameters[p]['features']
            feats=self.data_form.precompute_features(smiles=smiles,gen_features=feats,feature_generators=self.feature_generators,feature_list=features,original_indices=original_indices, indices=indices)
                
        nb_samples=feats[features[0]].shape[0]
        pred={}
        properties_empty_features={}
        for p in allprops:
            features =self.tasksfeatures_parameters[p]['features']
            
            flags_matrix = [[None] * len(features) for s in range(nb_samples)]
            for i, feature_name in enumerate(features):
                for j, row in enumerate(feats[feature_name]):
                    flags_matrix[j][i] = np.isnan(row).any()
                    if indices is None:
                        if list(smiles)[j]==None:
                            flags_matrix[j][i] = True
                        elif len(list(smiles)[j])<1:
                            flags_matrix[j][i] = True
                    if indices is not None:
                        ii,jj = indices[j]
                        if list(smiles)[ii]==None:
                            flags_matrix[j][i] = True
                        elif len(list(smiles)[ii])<1:
                            flags_matrix[j][i] = True
                        elif list(smiles)[jj]==None:
                            flags_matrix[j][i] = True
                        elif len(list(smiles)[jj])<1:
                            flags_matrix[j][i] = True
                        
            empty_features = [any(flags_row) for flags_row in flags_matrix]

            blender_props =self.tasksfeatures_parameters[p]['blender_properties']

            blender_l=[]
            for p_blen in blender_props:
                if p_blen in blender_properties_dict:
                    blender_l.append(blender_properties_dict[p_blen])
                else:
                    raise RuntimeError(f'{p_blen} not found in blender properties dictionary')

            if len(blender_l)>0:
                blender_l=[np.expand_dims(a, axis=1) if len(a.shape)<2 else a for a in blender_l] 
                X_blender_val=np.hstack(blender_l)
            else:
                X_blender_val=None

            
            empty_features_indexes = [i for i, is_feature_empty in enumerate(empty_features) if is_feature_empty]
            
            X_train=self.data_form.apply_feature_operation( [feats[k] for k in features])
            # If empty_features_indexes is an empty list, X_train will not be changed
            X_train[empty_features_indexes] = np.zeros(X_train.shape[1])
            if X_blender_val is not None:
                X_train=np.hstack((X_train,X_blender_val))
            if p.startswith(self.multiprop_prefix):
                pred,prop_keys=self.predict_multi_property(pred,X_train,p,compute_SD,empty_features_indexes,convert_log10)
            else:
                pred,prop_keys=self.predict_property(pred,X_train,p,compute_SD,empty_features_indexes,convert_log10)
            for k in prop_keys:
                properties_empty_features[k]=empty_features_indexes
            
        if len(smiles)==1:
            pred={ keys: value if type(value) is np.ndarray else np.array([value]) for keys,value in pred.items() }
        
        if convert_log10:
            logk= [k for k in pred.keys() if k.startswith('predicted_log10') ]
            logitk= [k for k in pred.keys() if k.startswith('predicted_logit') ]
            transformationsk= [k for k in self.transformations_dict.keys()]
            if len(logk):
                for p in logk:
                    tp='_'.join(p.split('_')[2:])
                    pred[f'predicted_{tp}']=10**pred[p]
                    pred.pop(p)
                    properties_empty_features[f'predicted_{tp}']=properties_empty_features[p]
                    properties_empty_features.pop(p)
            if len(logitk):
                for p in logitk:
                    tp='_'.join(p.split('_')[2:])
                    pred[f'predicted_{tp}']=1/(1+np.exp(-pred[p]))
                    pred.pop(p)
                    properties_empty_features[f'predicted_{tp}']=properties_empty_features[p]
                    properties_empty_features.pop(p)
            if len(transformationsk):
                for p in transformationsk:
                    pred[f'predicted_{p}']=self.transformations_dict[p](pred[f'predicted_{p}'])
        for property_name, prediction in pred.items():
            property_empty_features = properties_empty_features[property_name]
            prediction = prediction.astype(object)
            prediction[property_empty_features] = np.full(len(property_empty_features), None)
            pred[property_name] = prediction
            
        return { key:item.astype(dtype=float) if 'label' not in key else item for key,item in pred.items()}
    
    def predict_multi_property(self,pred,X_train,p,compute_SD,empty_features_indexes,convert_log10=False):
        keys=[]
        multi_prop=self.split_multi_property_name(p)
        is_model_list=isinstance(self.models[p],list)
        y_base_preds=None
        if is_model_list:
            y_preds_multi =np.array([m.predict(X_train).ravel(order='F') for m in self.models[p]])
        else:
            y_preds_multi=self.models[p].predict(X_train).ravel(order='F')
        if p in self.base_estimators:
            y_base_preds_multi =np.array([m.predict(X_train).ravel(order='F') for m in self.base_estimators[p]])
        nb_samples=X_train.shape[0]
        for i, p_i in enumerate(multi_prop):
            start=i*nb_samples
            end=(i+1)*nb_samples
            if is_model_list:
                y_preds=y_preds_multi[:,start:end]
                y_preds[:, empty_features_indexes] = np.full(len(empty_features_indexes), np.nan)
            else:
                y_preds=y_preds_multi[start:end]
                y_preds[empty_features_indexes] = np.nan
            if y_base_preds is not None:
                y_base_preds=y_base_preds_multi[:,start:end]
                y_base_preds[:, empty_features_indexes] = np.full(len(empty_features_indexes), np.nan)

            pred,keys_i=self.format_property_output(pred,p_i,compute_SD,empty_features_indexes, convert_log10,is_model_list=is_model_list, y_preds=y_preds, y_base_preds=y_base_preds)
            for k_i in keys_i:
                keys.append(k_i)
        return pred,keys
    
    def predict_property(self,pred,X_train,p,compute_SD,empty_features_indexes,convert_log10=False):
        """     
        predicts the output for one property for regression models
        
        Args:
             pred: output dictionary
             X_train: features of sample to be predicted
             p: property p
             compute_SD: compute standard_deviation
             empty_features_indexes: nan features
             convert_log10: boolean to revert transformer predictions
        
        Returns: 
            updated output dictionary and output keys
        """
        is_model_list=isinstance(self.models[p],list)
        y_base_preds=None
        if is_model_list:
            y_preds =np.array([m.predict(X_train).ravel() for m in self.models[p]])
            y_preds[:, empty_features_indexes] = np.full(len(empty_features_indexes), np.nan)
        else:
            y_preds=self.models[p].predict(X_train)
            y_preds[empty_features_indexes] = np.nan
        if p in self.base_estimators:
            y_base_preds =np.array([m.predict(X_train).ravel() for m in self.base_estimators[p]])
            y_base_preds[:, empty_features_indexes] = np.full(len(empty_features_indexes), np.nan)
            
        return self.format_property_output(pred,p,compute_SD,empty_features_indexes, convert_log10,is_model_list=is_model_list, y_preds=y_preds,y_base_preds=y_base_preds)
    
    def format_property_output(self,pred,p,compute_SD,empty_features_indexes,convert_log10=False,is_model_list=False,y_preds=None,y_base_preds=None):
        """     
        predicts the output for one property for regression models
        
        Args:
             pred: output dictionary
             X_train: features of sample to be predicted
             p: property p
             compute_SD: compute standard_deviation
             empty_features_indexes: nan features
             convert_log10: boolean to revert transformer predictions
        
        Returns: 
            updated output dictionary and output keys
        """
        keys=[]
        if is_model_list:
            #if y_pred.ndim> 1: y_pred=np.concatenate(y_pred , axis=0)
            pred[f'predicted_{p}']=np.mean(y_preds,axis=0)
        else:
            pred[f'predicted_{p}']=y_preds
        keys.append(f'predicted_{p}')
        if compute_SD:
            transform_log10=convert_log10 and p.startswith('log10')
            transform_logit=convert_log10 and p.startswith('logit')
            transform_dictionary=convert_log10 and p in self.transformations_dict
            if is_model_list:
                if transform_log10:
                    tp='_'.join(p.split('_')[1:])
                    pred[f'predicted_{tp}_std']=np.std(10**y_preds,axis=0)
                    keys.append(f'predicted_{tp}_std')
                elif transform_logit:
                    tp='_'.join(p.split('_')[1:])
                    pred[f'predicted_{tp}_std']=np.std(1/(1+np.exp(-y_preds)),axis=0)
                    keys.append(f'predicted_{tp}_std')
                elif transform_dictionary:
                    pred[f'predicted_{p}_std']=np.std(self.transformations_dict[p](y_preds),axis=0)
                    keys.append(f'predicted_{p}_std')
                else:
                    pred[f'predicted_{p}_std']=np.std(y_preds,axis=0)
                    keys.append(f'predicted_{p}_std')
            elif y_base_preds is not None:
                if transform_log10:
                    tp='_'.join(p.split('_')[1:])
                    pred[f'predicted_{tp}_std']=np.std(10**y_base_preds,axis=0)
                    keys.append(f'predicted_{tp}_std')
                elif transform_logit:
                    tp='_'.join(p.split('_')[1:])
                    pred[f'predicted_{tp}_std']=np.std(1/(1+np.exp(-y_base_preds)),axis=0)
                    keys.append(f'predicted_{tp}_std')
                else:
                    pred[f'predicted_{p}_std']=np.std(y_base_preds,axis=0)
                    keys.append(f'predicted_{p}_std')
        return pred,keys
    
    def delete_properties(self,properties):
        """    
        functionality to remove models/properties from the stacking model(s)
        
        Args:
             properties: list of properties to be removed
        """
        if not isinstance(properties,list):
            properties=[properties]
        for p in properties:
            if p in self.tasksfeatures_parameters:
                del self.tasksfeatures_parameters[p]
            if p in self.models:
                del self.models[p]
            if p in self.base_estimators:
                del self.base_estimators[p]
                
    def is_FeatureGenerationRegressor(self):
        """    
        Returns: 
            True
        """
        return True
    
    def merge_model(self,other_model,other_props=None):
        """    
        Merge properties from other model to this model
        
        Args:
             other_model: other stacking model
             other_props: properties to be merged of the other stacking model [=None], if None all properties of other model are merged
        """
        assert other_model.is_FeatureGenerationRegressor(), 'other model is not a FeatureGenerationRegressor'
        assert self.is_FeatureGenerationClassifier()==other_model.is_FeatureGenerationClassifier(), 'Trying to merge classifier with regressor, prediction output conflict, aborting... '
        if other_props==None:
            other_props=[p for p in other_model.models]
        elif not isinstance(other_props,list):
            other_props=[other_props]
        else:
            for p in other_props:
                assert p in other_model.models, f'other model has no property called {p}'
                        
        for key,method in other_model.feature_generators.items():
            if key not in self.feature_generators:
                self.feature_generators[key]=method
        
        for p in other_props:
                if p in other_model.tasksfeatures_parameters:
                    self.tasksfeatures_parameters[p]=other_model.tasksfeatures_parameters[p]
                if p in other_model.models:
                    self.models[p]=other_model.models[p]
                if p in other_model.base_estimators:
                    self.base_estimators[p]=other_model.base_estimators[p]
                self.merged_model=True
                
        if self.reproducable_output:
            print('regenerating reproducable output')
            self.generate_reproducable_output()
        
        
    ##########################################
    def validate(self,df, # df with smiles and the properties
                       props, # name of the task
                    true_props=None,# name of the property in df
                    smiles=None):
        """    
        [internal] Validate the found models
        
        Args:
             df: dataframe
             props: properties
             true_props: columns with true values
             smiles: smiles column
        """
        if not smiles: smiles=self.smiles
        if not props: allprops = [p for p in  self.models]
        
        elif isinstance(props, (list, tuple)) :
            for prop in props: assert prop in  self.models, f' property {prop} not in the dataset'
            allprops=props
        else:
            allprops=[props]
            
        if not true_props: alltrue=allprops
        elif isinstance(true_props, (list, tuple)) :
            if df:
                for prop in true_props: assert prop in df.columns
            alltrue=true_props
        else:
            alltrue=[true_props]
        
        
        assert len(alltrue)== len(allprops)
        nav=[]
        if isinstance(df, type(None)) and  isinstance(self.Validation, pd.DataFrame):
            if self.verbose: print( 'Using the validation dataset which is stored in the model')
                
            na=self.Validation[smiles].isna()
            if(sum(na)):
                if self.verbose: 
                    print(f'  Removing the following {sum(na)}nan smiles')
                    print(self.Validation[~na,smiles])  
                self.Validation=self.Validation[~na]
            df= self.Validation
            for prop in allprops: 
                #assert prop in self.Validation.columns
                features =self.tasksfeatures_parameters[prop]['features'] 
                self.original_indices['Validation'],self.pair_indices['Validation']=self.data_form.get_pairs(self.Validation,pairs_col='pairs')
                if not self.FEATURES['Validation']: self.FEATURES['Validation']={}
                self.FEATURES['Validation']=self.data_form.precompute_features(smiles=self.Validation[smiles],gen_features=self.FEATURES['Validation'], feature_generators=self.feature_generators, feature_list=features, original_indices=self.original_indices['Validation'], indices=self.pair_indices['Validation'])
            
            y_preds={}
            y_true_d={}
            for i,p in enumerate(allprops):
                blender_properties =self.tasksfeatures_parameters[p]['blender_properties']   
                features =self.tasksfeatures_parameters[p]['features']
                X_valid,y_valid,nav_p= self.data_form.create_X_y(self.Validation,p, smiles,self.FEATURES['Validation'],features, p.startswith(self.multiprop_prefix),indices=self.pair_indices['Validation'])
                #X_valid,y_valid,nav_p= self._create_X_y(self.Validation,p, 'Validation',smiles,self.FEATURES['Validation'],features)
                blender_l=[]
                for p_blen in blender_properties:
                    assert p_blen in self.Train.columns
                    _,X_blender_i,na_i= self.data_form.create_X_y(self.Validation,p_blen, smiles,self.FEATURES['Validation'],features, False,indices=self.pair_indices['Validation'])
                    blender_l=[b[~na_i[~nav_p]] for b in blender_l]
                    blender_l.append(X_blender_i[~nav_p[~na_i]])
                    X_valid=X_valid[~na_i[~nav_p]]
                    y_valid=y_valid[~na_i[~nav_p]]
                    nav_p=np.logical_or(nav_p,na_i)
                if len(blender_l)>0:
                    blender_l=[np.expand_dims(a, axis=1) if len(a.shape)<2 else a for a in blender_l] 
                    X_blender_val=np.hstack(blender_l)
                else:
                    X_blender_val=None

                if X_blender_val is not None:
                    X_valid=np.hstack((X_valid,X_blender_val))

                y_true_d[p]=y_valid
                nav.append(nav_p)
                if p.startswith(self.multiprop_prefix):
                    y_preds,_=self.predict_multi_property(y_preds,X_valid,p,self.compute_SD,[])
                else:
                    y_preds,_s=self.predict_property(y_preds,X_valid,p,self.compute_SD,[])
        else:
            SM=df.loc[df[smiles].notnull(),smiles]
            y_preds=self.predict(props=allprops, smiles=SM)
        validation_props=[]
        for ip,p in enumerate(alltrue):
            if p.startswith(self.multiprop_prefix):
                nav_i=nav[ip]
                multi_prop=self.split_multi_property_name(p)
                for it in range(len(multi_prop)-1):
                    nav.insert(ip, nav_i)
                for p_i in multi_prop:
                    assert p_i in self.Validation.columns
                    validation_props.append(p_i)
            else:
                assert p in self.Validation.columns
                validation_props.append(p)
        self.show_results(validation_props,y_preds,validation_props,df,nav=nav,y_true_d=y_true_d)
    ##########################################
    #show regression results
    def show_results(self,allprops,y_preds,alltrue,df,nav=None,y_true_d={}):
        """    
        [internal] shows result for the properties
        
        Args:
             allprops: the properties
             y_preds: the dictionary containing the predictions
             alltrue: the values of the true values
             df: dataframe
        """
        nr=len(allprops)
        if nr==1:
            ncols=1
            nrows=1
        else:
            nrows=math.ceil(nr/2)
            ncols=2
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols ,4*nrows))
        i=0
        while i < len(allprops):
            y_pred=y_preds[f'predicted_{allprops[i]}']
            if f'predicted_{allprops[i]}_std' in y_preds:
                 yerr=y_preds[f'predicted_{allprops[i]}_std']
            tp=alltrue[i]
            if nav is None:
                mask=df[tp].isna()
                y_true=df[tp].values
            else:
                if tp in y_true_d:
                    y_true=y_true_d[tp]
                    mask=np.isnan(y_true)
                else:
                    mask=df[tp][~nav[i]].isna()
                    y_true=df[tp][~nav[i]].values
            if nr>1: ax=axs.flat[i]
            else: ax=axs
            plot_reg_model(y_pred[~mask],  y_true[~mask],title=f'Validation {tp}', ax=ax)
            if f'predicted_{allprops[i]}_std' in y_preds:
                ax.errorbar(x= y_true[~mask], y=y_pred[~mask], yerr= yerr[~mask],fmt='none' ,alpha=0.2,color='red')
            i +=1
        if i > 1:
            while i < ncols*nrows:
                axs.flat[i].set_axis_off()
                i+=1
        fig.tight_layout()
        fig.show()
        
    ##########################################
        
    def print_metrics(self):
        """    
        prints the model and results
        
        Returns: 
            list of strings of the printed models
        """
        if self.merged_model:
            output_str='Merged model, print statement disabled'
            print(output_str)
            return output_str
        else:
            CEND      = '\33[0m'
            CGREEN  = '\33[32m'
            CYELLOW = '\33[33m'
            CBOLD     = '\33[1m'
            output_str=[]
            for p in self.models:
                if isinstance(self.models[p],list):
                    models='\n'.join([str(m.steps[1][1])+':'+str(m.steps[2][1]) if m.steps[1][1]!='passthrough' else str(m.steps[2][1]) for m in self.models[p]]  )
                else:
                    m=self.models[p]
                    models='\n'.join([str(m.steps[1][1])+': '+str(m.steps[2][1]) if m.steps[1][1]!='passthrough' else str(m.steps[2][1])] )
                m=self.metrics[p]
                print(CBOLD+p+CEND ,':', m)
                print('List of models:')
                print(models)
                output_str.append(models)
            return output_str
            
    #####################################
    def __call__(self):
        pass
    def clean(self):
        """    
        removes features and groups
        """
        self.FEATURES={'Train': {}, 'Validation':{}}
        self.groups=None
        self.Scaffold=None
        
    def deep_clean(self):
        """    
        Clean the model for deployment, clears data and features
        """
        self.clean()
        self.Train = self.Train.iloc[0:0]
        self.Validation = self.Validation.iloc[0:0]
    ##########################################
    def Data_clustering(self,method= "Bottleneck", # Bottleneck or Butina or Scaffold
                        n_groups=5 , # for Bottleneck
                        cutoff=0.6,  # for Butina
                        include_chirality=False,
                       random_state=42,
                       clustering_algorithm:ClusteringAlgorithm =None
                       ):#,n_jobs=-1):
        '''
        perform clustering of the smiles to get groups for splitting
        
        method: Bottleneck       : kmeans using Bottleneck feautures
                Butina   : Butina clustering using  cutoff
                Scaffold : get Murcko Scaffold using include_chirality
        
        Args:        
             method: clustering method as string ['Bottleneck','Butina','HierarchicalButina', 'Scaffold']
             n_groups: number of k for kmeans
             cutoff: cutoff for Butina or list of cutoffs for HierarchicalButina
             include_chirality: boolean for Scaffold
             random_state: integer
             clustering_algorithm: provided clustering algorithm, overwrites given string for method
        '''
        if clustering_algorithm is None: 
            assert method in ['Bottleneck','Butina','HierarchicalButina', 'Scaffold'], 'provide one of Bottleneck or Butina or Scaffold'
            if method== 'Scaffold':
                clustering_algorithm=MurckoScaffoldClustering(include_chirality=include_chirality)
            elif method== 'Butina':
                clustering_algorithm=ButinaSplitReassigned(cutoff = cutoff)
            elif method=='HierarchicalButina':
                clustering_algorithm=HierarchicalButina(cutoff = cutoff)
            elif method== 'Bottleneck':
                clustering_algorithm=KmeansForSmiles(n_groups=n_groups,feature_generators=self.feature_generators,used_features=['Bottleneck'],random_state=random_state)
                if self.verbose: 
                    print(f'Perform km clustering ({n_groups} clusters) using the Bottleneck features' )
                
        assert isinstance(self.Train, pd.DataFrame) and  self.smiles in self.Train.columns, 'Check train dataset and the smiles column'
        clustering_algorithm.cluster(self.Train[self.smiles])
        self.groups=clustering_algorithm.get_groups()
        generated_feat=clustering_algorithm.get_generated_features()
        #copy features
        for key,item in generated_feat.items():
            if key in self.feature_generators:
                if item['cid']==self.feature_generators[key].get_generator_name():
                    if not self.FEATURES['Train']: self.FEATURES['Train']={}
                    self.FEATURES['Train'][key]=item['X'].copy()
                    
        clustering_algorithm.clear_generated_features()            
        self.clusters=pd.DataFrame({'groups':self.groups}).groupby(by=['groups']).indices
        if self.verbose:
            print('Number of groups=', len(np.unique(self.groups)))
            print(pd.DataFrame({'groups':self.groups})['groups'].value_counts())#'culusters count:\n',Counter(self.groups))#
            
            
########################################################
class FeatureGenerationRegressionClassifier(FeatureGenerationRegressor):
    """    
    FeatureGenerationClassifier is a specialization of FeatureGenerationRegressor for RegressionClassification
    """
    
    def __init__(self,model=None, use_gpu=False,compute_SD=False,labelnames=None,feature_generators=None, outer_jobs=None,verbose=False,relative_modelling=False,
                            feature_operation:str='concat',property_operation='identical' ):
        '''
        Initialization
        
        Args:
             model: string indicating encoder
             use_gpu: boolean
             compute_SD: boolean to compute standard deviation of output
             labelnames: labelnames of the classes
             feature_generators: dictionary with feature_generators [=None]
             outer_jobs: number of threads for outer cross-validation
             verbose
        '''
        super().__init__(model=model, use_gpu=use_gpu,compute_SD=compute_SD,feature_generators=feature_generators, outer_jobs=outer_jobs,verbose=verbose, relative_modelling=relative_modelling,feature_operation=feature_operation, property_operation=property_operation)
        self.th_dict={}
        for key,val in labelnames.items():
            assert len(val)==2, 'non-binary labelnames given, make sure that the number of classes is two for regressionclassification'
        ## labelnames of the classes (dictionary)
        self.labelnames=labelnames
    
    def predict_empty_smiles(self,allprops,compute_SD=False,convert_log10=True):
        """    
        creates empty output but with the right format
        
        Args:
             allprops: the properties
             compute_SD: add standard deviation
             convert_log10: boolean to revert to original values
        
        Returns: 
            dictionary with all the keys but empty arrays
        """
        pred={}
        for p in allprops:
            pred[f'predicted_{p}']=np.array([],dtype=object)
            pred[f'predicted_labels_{p}']=np.array([],dtype=object)
            for c in range(2):
                pred[f'predicted_proba_{p}_class_{c}']=np.array([],dtype=object)
            if compute_SD:
                pred[f'predicted_{p}_ratio']=np.array([],dtype=object)
                for c in range(2):
                    pred[f'predicted_proba_{p}_class_{c}_std']=np.array([],dtype=object)
        return pred
    
    def set_property_threshold(self,prop,threshold=0.5):
        """    
        Set property threshold for a specific class. 
            
        If the probability prediction for that class is larger than the threshold, the sample is assigned to this class.
        The remaining samples are asigned using argmax on the remaining probability predictions for the other classes. Threshold optimization as one versus others.  
        
        Args:
             prop: property/model for which the trheshold has to be set
             threshold: the threshold to be set 
        """
        assert threshold>=0 and threshold<=1.0, 'given threshold is not between 0 and 1'
        self.th_dict[prop]=(1,threshold)
    
    
    def predict_multi_property(self,pred,X_train,p,compute_SD,empty_features_indexes,convert_log10=False):
        keys=[]
        multi_prop=self.split_multi_property_name(p)
        if isinstance(self.models[p],list):
            temp_y =np.array([m.predict(X_train).ravel(order='F') for m in self.models[p]])
            #temp_y[:, empty_features_indexes] = np.full(len(empty_features_indexes), np.nan)
            #if y_pred.ndim> 1: y_pred=np.concatenate(y_pred , axis=0)            
            y_pred_proba_multi =np.mean(temp_y,axis=0)
        else:
            y_pred_proba_multi =self.models[p].predict(X_train).ravel(order='F')

        y_base_preds=None
        if compute_SD:
            if p in self.base_estimators or isinstance(self.models[p],list):
                if isinstance(self.models[p],list):
                     model_list=self.models[p]
                else:
                     model_list=self.base_estimators[p]
                y_base_preds_multi =[m.predict(X_train).ravel(order='F') for m in model_list]
        
        nb_samples=X_train.shape[0]
        for i, p_i in enumerate(multi_prop):
            start=i*nb_samples
            end=(i+1)*nb_samples
            y_pred_proba=y_pred_proba_multi[start:end]
            y_pred_proba[empty_features_indexes] = np.nan
            if y_base_preds is not None:
                y_base_preds=[y_base[start:end] for y_base in y_base_preds_multi]
                for y_base in y_base_preds:
                    y_base[empty_features_indexes] = np.nan

            pred,keys_i=self.format_property_output(pred,p_i,compute_SD,empty_features_indexes, convert_log10,is_model_list=False, y_preds=y_pred_proba,y_base_preds=y_base_preds)
            for k_i in keys_i:
                keys.append(k_i)
        return pred,keys
    

    def predict_property(self,pred,X_train,p,compute_SD,empty_features_indexes,convert_log10=False):
        """    
        specialization of FeatureGenerationRegressor.predict_property for regressionclassification
        
        Args:
             pred: output dictionary
             X_train: features of sample to be predicted
             p: property p
             compute_SD: compute standard_deviation
             empty_features_indexes: nan features
             convert_log10: boolean to revert transformer predictions
        
        Returns: 
            updated output dictionary and output keys
        """
        if isinstance(self.models[p],list):
            temp_y =np.array([m.predict(X_train).ravel() for m in self.models[p]])
            temp_y[:, empty_features_indexes] = np.full(len(empty_features_indexes), np.nan)
            #if y_pred.ndim> 1: y_pred=np.concatenate(y_pred , axis=0)            
            y_pred_proba =np.mean(temp_y,axis=0)
        else:
            y_pred_proba =self.models[p].predict(X_train)

        y_base_preds=None
        if compute_SD:
            if p in self.base_estimators or isinstance(self.models[p],list):
                if isinstance(self.models[p],list):
                     model_list=self.models[p]
                else:
                     model_list=self.base_estimators[p]
                y_base_preds =[m.predict(X_train).ravel() for m in model_list]
                
        return self.format_property_output(pred,p,compute_SD,empty_features_indexes, convert_log10,is_model_list=False, y_preds=y_pred_proba, y_base_preds=y_base_preds)
    
    def format_property_output(self,pred,p,compute_SD, empty_features_indexes, convert_log10=False,is_model_list=False,y_preds=None,y_base_preds=None):
        """    
        specialization of FeatureGenerationRegressor.predict_property for regressionclassification
        
        Args:
             pred: output dictionary
             X_train: features of sample to be predicted
             p: property p
             compute_SD: compute standard_deviation
             empty_features_indexes: nan features
             convert_log10: boolean to revert transformer predictions
        
        Returns: 
            updated output dictionary and output keys
        """
        keys=[]        
        th=0.5
        c=1
        if p in self.th_dict:
            c,th=self.th_dict[p]

        y_pred_proba=np.clip(y_preds, 0.0, 1.0)
        y_preds=np.full(shape=y_pred_proba.shape,fill_value=0,dtype=np.int64)
        th_mask=y_pred_proba>th
        y_preds[th_mask]=c
        y_preds[~th_mask]=0
            
        y_preds[empty_features_indexes] = -1
        y_pred_proba[empty_features_indexes] = np.nan
        pred[f'predicted_{p}']=y_preds
        pred[f'predicted_labels_{p}']=np.array([str(self.labelnames[p][c]) if c>=0 else str('NaN') for c in y_preds])
        keys.append(f'predicted_{p}')
        keys.append(f'predicted_labels_{p}')
        
        pred[f'predicted_proba_{p}_class_{self.labelnames[p][0]}']=1-y_pred_proba
        pred[f'predicted_proba_{p}_class_{self.labelnames[p][1]}']=y_pred_proba
        keys.append(f'predicted_proba_{p}_class_{self.labelnames[p][0]}')
        keys.append(f'predicted_proba_{p}_class_{self.labelnames[p][1]}')

        if compute_SD:
            if y_base_preds is not None:
                y_base_preds_proba_c =np.array(y_base_preds)
                pred[f'predicted_proba_{p}_class_{self.labelnames[p][0]}_std']=np.std(y_base_preds_proba_c,axis=0)
                keys.append(f'predicted_proba_{p}_class_{self.labelnames[p][0]}_std')
                pred[f'predicted_proba_{p}_class_{self.labelnames[p][1]}_std']=pred[f'predicted_proba_{p}_class_{self.labelnames[p][0]}_std']
                keys.append(f'predicted_proba_{p}_class_{self.labelnames[p][1]}_std')
                
                for m in range(len(y_base_preds)):
                    base_mask=y_base_preds[m]>th
                    y_base_preds[m][base_mask]=c
                    y_base_preds[m][~base_mask]=0
                    y_base_preds[m]=y_base_preds[m]==y_preds
                y_base_preds=np.array(y_base_preds)
                y_base_preds[:,empty_features_indexes] = np.full(len(empty_features_indexes), np.nan)
                pred[f'predicted_{p}_ratio']=np.sum(y_base_preds,axis=0)/len(y_base_preds)
                keys.append(f'predicted_{p}_ratio')
            
        return pred,keys

                
    def is_FeatureGenerationRegressionClassifier(self):
        """    
        Returns: 
            True
        """
        return True
    
    def merge_model(self,other_model,other_props=None):
        """    
        Merge properties from other model to this model
        
        Args:
             other_model: other stacking model
             other_props: properties to be merged of the other stacking model [=None], if None all properties of other model are merged
        """
        assert other_model.is_FeatureGenerationRegressionClassifier(), 'other model is not a is_FeatureGenerationClassifier'
        if other_props==None:
            other_props=props=[p for p in other_model.models]
        elif not isinstance(other_props,list):
            other_props=[other_props]
        else:
            for p in other_props:
                assert p in other_model.models, f'other model has no property called {p}'
                
        for p in other_props:
            if p in other_model.th_dict:
                self.th_dict[p]=other_model.th_dict[p]
            if p in other_model.labelnames:
                self.labelnames[p]=other_model.labelnames[p]
        
        #at the end, since generate_reproducable_output may be executed.
        super().merge_model(other_model,other_props=other_props)
    
    def delete_properties(self,properties):
        """    
        functionality to remove models/properties from the stacking model(s)
        
        Args:
             properties: list of properties to be removed
        """
        if not isinstance(properties,list):
            properties=[properties]
        super().delete_properties(properties)
        for p in properties:
            if p in self.th_dict:
                del self.th_dict[p]
            if p in self.labelnames:
                del self.labelnames[p]
    
        ###########################################  
    #show classication results
    def show_results(self,allprops,y_preds,alltrue,df,nav=None,y_true_d={}):
        """    
        [internal] shows result for the properties
        
        Args:
             allprops: the properties
             y_preds: the dictionary containing the predictions
             alltrue: the values of the true values
             df: dataframe
        """
        i=0
        while i < len(allprops):
            y_pred=y_preds[f'predicted_{allprops[i]}']
            y_pred_proba=np.concatenate(tuple(np.expand_dims(y_preds[f'predicted_proba_{allprops[i]}_class_{self.labelnames[allprops[i]][c]}'],axis=1) for c in range(2)), axis=1)
            
            tp=alltrue[i]
            if nav is None:
                mask=df[tp].isna()
                y_true=df[tp].values
            else:
                if tp in y_true_d:
                    y_true=y_true_d[tp]
                    mask=np.isnan(y_true)
                else:
                    mask=df[tp][~nav[i]].isna()
                    y_true=df[tp][~nav[i]].values
            print(f'predicted_{allprops[i]}, mean roc auc: ', roc_auc_score(y_true[~mask]==1, y_pred_proba[~mask,1]))
            print(classification_report(y_true[~mask], y_pred[~mask],labels=[ index for index in range(y_pred_proba.shape[1])], target_names= list(self.labelnames[allprops[i]].values())))
            i+=1
            
    
    
########################################################

class FeatureGenerationClassifier(FeatureGenerationRegressor):
    """
    FeatureGenerationClassifier is a specialization of FeatureGenerationRegressor for classification
    """
    
    def __init__(self,model=None, use_gpu=False,compute_SD=False,labelnames=None,feature_generators=None, outer_jobs= None,verbose=False,relative_modelling=False,
                            feature_operation:str='concat',property_operation='identical'):
        '''
        Initialization
        
        Args:
             model: string indicating encoder
             use_gpu: boolean
             compute_SD: boolean to compute standard deviation of output
             labelnames: labelnames of the classes
             feature_generators: dictionary with feature_generators [=None]
             outer_jobs: number of threads for outer cross-validation
             verbose
        '''
        super().__init__(model=model, use_gpu=use_gpu,compute_SD=compute_SD,feature_generators=feature_generators, outer_jobs=outer_jobs,verbose=verbose,relative_modelling=relative_modelling,feature_operation=feature_operation, property_operation=property_operation)
        self.nb_classes={}
        self.th_dict={}
        self.labelnames=labelnames
    
    def create_modfinder(self):
        """    
        Returns the appropriate derivation of ModelFinder
        
        Returns: ClassificationFinder(verbose,NestedCVModelSearch,outer_jobs)
        """
        return ClassificationFinder(self.verbose,NestedCVModelSearch(self.verbose),outer_jobs=self.outer_jobs)
    
    ###########################################
    #search and store found classifier(s)
    def do_kfold_search(self,prop,X, y,X_blender, groups,params_grid,paramsearch, scoring ,cv=3 ,n_jobs=-1,n_iter=None ,use_memory=False, outer_cv_fold=5, split='GKF',blender_params=None,prefix_dict=None,random_state=42,sample_weight=None):
        """    
        specialization of FeatureGenerationRegressor.do_kfold_search for classification
        
        If no top estimator, then the the predictions of the models are given to a voting classifier
        else the output is given to the top estimator
        
        Args:
             prop: the property/column of df with the target values
             X: the feature matrix
             y: the target values corresponding to the samples of X
             groups: the non-overlapping groups
             params_grid: the base estimator parameter grid
             scoring: string representing scoring function used in scikit-learn (default:'r2')
             cv: number of folds for inner cross-validation (default:3)
             outer_cv_fold: number of folds for outer cross-validation (default:4)
             n_jobs: number of threads used (default:-1)
             use_memory: use memory in pipeline (default:False)
             split: string to select options from different splits for cross-validation (default:GKF)
             n_iter: number of iterations randomized_search, if None grid_search is performed(default:None)
             blender_params: the top estimator parameter grid
             prefix_dict: dictionary of prefix strings used in the paramter grids
             random_state: random state initialization value
             sample_weight: sample weights
        """
        if prop.startswith(self.multiprop_prefix):
            multi_prop=self.split_multi_property_name(prop)
            for i,single_p in enumerate(multi_prop):
                self.nb_classes[single_p]=int(np.nanmax(y[:,i])+1)
        else:
            self.nb_classes[prop]=int(np.nanmax(y)+1)
        modfinder=self.create_modfinder()
        CV= modfinder.model_search( X=X, y=y,X_blender=X_blender, groups=groups,params_grid=params_grid,paramsearch=paramsearch, scoring=scoring ,cv=cv ,
                                use_memory=False,verbose=self.verbose, outer_cv_fold=outer_cv_fold
                                ,split=split,blender_params=blender_params,prefix_dict=prefix_dict,random_state=random_state,sample_weight=sample_weight)
        if blender_params is not None and CV['blender_model'] is not None:
            #copy base estimators from stackingClassifier
            if isinstance(CV['blender_model'],  type(StackingClassifier([None]))):
                self.models[prop]= clone(CV['blender_model']).fit(X,y)
                self.base_estimators[prop] =[clone(m[1]).fit(X,y) for m in self.models[prop].estimators]
            #get model_list from BaseEstimatorTransformer
            else:
                if sample_weight is not None:
                    fit_params={}
                    est_prefix=prefix_dict['method_prefix']
                    red_prefix=prefix_dict['dim_prefix']
                    fit_params[f'{red_prefix}__{est_prefix}__sample_weight']=sample_weight
                    fit_params[f'{est_prefix}__sample_weight']=sample_weight
                    self.models[prop]= clone(CV['blender_model']).fit(X,y,**fit_params)
                    del fit_params[f'{red_prefix}__{est_prefix}__sample_weight']
                    if hasattr(self.models[prop].steps[1][1], 'model_list'):
                        self.base_estimators[prop] =[clone(m).fit(X,y,**fit_params) for m in self.models[prop].steps[1][1].model_list]
                else:
                    self.models[prop]= clone(CV['blender_model']).fit(X,y)
                    if hasattr(self.models[prop].steps[1][1], 'model_list'):
                        self.base_estimators[prop] =[clone(m).fit(X,y) for m in self.models[prop].steps[1][1].model_list]
        else:
            self.models[prop]= VotingClassifier(estimators= [(f'method_{index}',clone(m)) for index,m in enumerate(CV['models'])],voting='soft')
            if sample_weight is not None:
                fit_params={}
                est_prefix=prefix_dict['method_prefix']
                fit_params[f'{est_prefix}__sample_weight']=sample_weight
                self.base_estimators[prop] =[clone(m[1]).fit(X,y,**fit_params) for m in self.models[prop].estimators]
                self.models[prop].fit(X,y)
            else:
                self.base_estimators[prop] =[clone(m[1]).fit(X,y) for m in self.models[prop].estimators]
                self.models[prop].fit(X,y)
        #self.models[prop]= [m for m in CV['models']]
        self.metrics[prop]=CV['Nested_CV score']
    ###########################################
    def predict_empty_smiles(self,allprops,compute_SD=False,convert_log10=True):
        """    
        creates empty output but with the right format
        
        Args:
             allprops: the properties
             compute_SD: add standard deviation
             convert_log10: boolean to revert to original values
        Returns: 
            dictionary with all the keys but empty arrays
        """
        pred={}
        props=[]
        for p in allprops:
            if p.startswith(self.multiprop_prefix):
                multi_prop=self.split_multi_property_name(p)
                for mp in multi_prop:
                    props.append(mp)  
            else:
                props.append(p)
        for p in props:
            pred[f'predicted_{p}']=np.array([],dtype=object)
            pred[f'predicted_labels_{p}']=np.array([],dtype=object)
            for c in range(self.nb_classes[p]):
                pred[f'predicted_proba_{p}_class_{c}']=np.array([],dtype=object)
            if compute_SD:
                pred[f'predicted_{p}_ratio']=np.array([],dtype=object)
                for c in range(self.nb_classes[p]):
                    pred[f'predicted_proba_{p}_class_{c}_std']=np.array([],dtype=object)
        return pred
    
    def set_property_threshold(self,prop,class_index=0,threshold=0.5):
        """    
        Set property threshold for a specific class. 
            
        If the probability prediction for that class is larger than the threshold, the sample is assigned to this class.
        The remaining samples are asigned using argmax on the remaining probability predictions for the other classes. Threshold optimization as one versus others.  
        
        Args:
             prop: property/model for which the trheshold has to be set
             threshold: the threshold to be set 
        """
        assert prop in self.nb_classes, f'given property {prop} is not seen by the model'
        assert class_index>=0 and class_index<self.nb_classes[prop], f'given class index is smaller than zero or larger than number of classes for property {prop}'
        assert threshold>=0 and threshold<=1.0, 'given threshold is not between 0 and 1'
        self.th_dict[prop]=(class_index,threshold)
    
    def predict_multi_property(self,pred,X_train,p,compute_SD,empty_features_indexes,convert_log10=False):
        keys=[]
        multi_prop=self.split_multi_property_name(p)
        
        y_preds =self.models[p].predict(X_train).ravel(order='F')
        y_preds_prob_multi =self.models[p].predict_proba(X_train)

        y_base_preds_multi=None
        y_base_preds_proba_multi=None
        if compute_SD:
            if p in self.base_estimators:
                y_base_preds_multi =[m.predict(X_train).ravel(order='F') for m in self.base_estimators[p]]
                #y_base_preds[:, empty_features_indexes] = np.full(len(empty_features_indexes), np.nan)
                #TODO: more efficient
                y_base_preds_proba_multi=[m.predict_proba(X_train) for m in self.base_estimators[p]]
                #for c,name in enumerate(self.labelnames[multi_prop[0]].values()):
                #    y_base_preds_proba_multi.append(np.array([np.concatenate(m.predict_proba(X_train),axis=0)[:,c].ravel(order='F') for m in self.base_estimators[p]]))
                    #y_base_preds_proba[c][:, empty_features_indexes] = np.full(len(empty_features_indexes), np.nan)
        
        nb_samples=X_train.shape[0]
        for i, p_i in enumerate(multi_prop):
            start=i*nb_samples
            end=(i+1)*nb_samples
            y_pred_proba=y_preds_prob_multi[i]
            y_pred_proba[empty_features_indexes] = np.nan
            if y_base_preds_multi is not None:
                y_base_preds=np.array([y_base[start:end]==y_preds[start:end] for y_base in y_base_preds_multi])
                for y_base in y_base_preds:
                    y_base[empty_features_indexes] = np.nan
                
                y_base_preds_proba=[]
                for c,name in enumerate(self.labelnames[p_i].values()):
                    y_base_preds_proba.append(np.concatenate([np.expand_dims(y_base_preds_proba_multi[m][i][:,c],axis=1) for m in range(len(y_base_preds_proba_multi))], axis=1).transpose())
                    y_base_preds_proba[c][:,empty_features_indexes] =np.full(len(empty_features_indexes), np.nan)
            pred,keys_i=self.format_property_output(pred,p_i,compute_SD,empty_features_indexes, convert_log10,is_model_list=False, y_preds_prob=y_pred_proba,y_base_preds=y_base_preds, y_base_preds_proba=y_base_preds_proba)
            for k_i in keys_i:
                keys.append(k_i)
        return pred,keys
    
    
    #predict the classes for one property
    def predict_property(self,pred,X_train,p,compute_SD,empty_features_indexes,convert_log10=False):
        """    
        specialization of FeatureGenerationRegressor.predict_property for classification
        
        Args:
             pred: output dictionary
             X_train: features of sample to be predicted
             p: property p
             compute_SD: compute standard_deviation
             empty_features_indexes: nan features
             convert_log10: boolean to revert transformer predictions
        
        Returns: 
            updated output dictionary and output keys
        """
        y_preds =self.models[p].predict(X_train)
        y_preds_prob =self.models[p].predict_proba(X_train)
        y_preds_prob[empty_features_indexes,:] = np.nan
        
        y_base_preds=None
        y_base_preds_proba=None
        if compute_SD:
            if p in self.base_estimators:
                y_base_preds =np.array([m.predict(X_train).ravel()==y_preds for m in self.base_estimators[p]])
                y_base_preds[:, empty_features_indexes] = np.full(len(empty_features_indexes), np.nan)
                #TODO: more efficient
                y_base_preds_proba=[]
                for c,name in enumerate(self.labelnames[p].values()):
                    y_base_preds_proba.append(np.array([m.predict_proba(X_train)[:,c].ravel() for m in self.base_estimators[p]]))
                    y_base_preds_proba[c][:, empty_features_indexes] = np.full(len(empty_features_indexes), np.nan)
                    
        return self.format_property_output(pred,p,compute_SD,empty_features_indexes,convert_log10=False,is_model_list=False, y_preds_prob=y_preds_prob, y_base_preds=y_base_preds, y_base_preds_proba=y_base_preds_proba)
    
    def format_property_output(self,pred,p,compute_SD,empty_features_indexes,convert_log10=False,is_model_list=False, y_preds_prob=None, y_base_preds=None, y_base_preds_proba=None):
        keys=[]
        y_preds =y_preds_prob.argmax(1)
        
        if p in self.th_dict:
            c,th=self.th_dict[p]
            y_prob_copy=y_preds_prob.copy()
            th_mask=y_prob_copy[:,c]>th
            y_preds[th_mask]=c
            y_prob_copy[:,c]=-1
            y_preds[~th_mask]=y_prob_copy[~th_mask,:].argmax(1)
            
        y_preds[empty_features_indexes] = -1
        y_preds_prob[empty_features_indexes,:] = np.nan
        pred[f'predicted_{p}']=y_preds
        pred[f'predicted_labels_{p}']=np.array([str(self.labelnames[p][c]) if c>=0 else str('NaN') for c in y_preds])
        keys.append(f'predicted_{p}')
        keys.append(f'predicted_labels_{p}')
        for c,name in enumerate(self.labelnames[p].values()):
            pred[f'predicted_proba_{p}_class_{name}']=y_preds_prob[:,c]
            keys.append(f'predicted_proba_{p}_class_{name}')
        if compute_SD:
            if y_base_preds is not None:
                pred[f'predicted_{p}_ratio']=np.sum(y_base_preds,axis=0)/len(y_base_preds)
                keys.append(f'predicted_{p}_ratio')
                #TODO: more efficient
                for c,name in enumerate(self.labelnames[p].values()):
                    y_base_preds_proba_c=y_base_preds_proba[c]
                    pred[f'predicted_proba_{p}_class_{name}_std']=np.std(y_base_preds_proba_c,axis=0)
                    keys.append(f'predicted_proba_{p}_class_{name}_std')
            
        return pred,keys
    
    def delete_properties(self,properties):
        """    
        functionality to remove models/properties from the stacking model(s)
        
        Args:
             properties: list of properties to be removed
        """
        if not isinstance(properties,list):
            properties=[properties]
        super().delete_properties(properties)
        for p in properties:
            if p in self.th_dict:
                del self.th_dict[p]
            if p in self.labelnames:
                del self.labelnames[p]
            if p in self.nb_classes:
                del self.nb_classes[p]
                
    def is_FeatureGenerationClassifier(self):
        """    
        Returns: 
            True
        """
        return True
    
    
    def merge_model(self,other_model,other_props=None):
        """    
        Merge properties from other model to this model
        
        Args:
             other_model: other stacking model
             other_props: properties to be merged of the other stacking model [=None], if None all properties of other model are merged
        """
        assert other_model.is_FeatureGenerationClassifier(), 'other model is not a is_FeatureGenerationClassifier'
        if other_props==None:
            other_props=props=[p for p in other_model.models]
        elif not isinstance(other_props,list):
            other_props=[other_props]
        else:
            for p in other_props:
                assert p in other_model.models, f'other model has no property called {p}'
                
        for p in other_props:
            if p in other_model.th_dict:
                self.th_dict[p]=other_model.th_dict[p]
            if p in other_model.labelnames:
                self.labelnames[p]=other_model.labelnames[p]
            if p in other_model.nb_classes:
                self.nb_classes[p]=other_model.nb_classes[p]
        
        #at the end, since generate_reproducable_output may be executed.
        super().merge_model(other_model,other_props=other_props)


            
    
    ###########################################  
    #show classication results
    def show_results(self,allprops,y_preds,alltrue,df,nav=None, y_true_d={}):
        """    
        [internal] shows result for the properties
        
        Args:
             allprops: the properties
             y_preds: the dictionary containing the predictions
             alltrue: the values of the true values
             df: dataframe
        """
        i=0
        while i < len(allprops):
            y_pred=y_preds[f'predicted_{allprops[i]}']
            y_pred_proba=np.concatenate(tuple(np.expand_dims(y_preds[f'predicted_proba_{allprops[i]}_class_{self.labelnames[allprops[i]][c]}'],axis=1) for c in range(self.nb_classes[allprops[i]])), axis=1)

            tp=alltrue[i]
            if nav is None:
                mask=df[tp].isna()
                y_true=df[tp].values
            else:
                if tp in y_true_d:
                    y_true=y_true_d[tp]
                    mask=np.isnan(y_true)
                else:
                    mask=df[tp][~nav[i]].isna()
                    y_true=df[tp][~nav[i]].values
            if self.nb_classes[allprops[i]]>2:
                print(f'predicted_{allprops[i]}, mean one vs one auc: ', roc_auc_score(y_true[~mask], y_pred_proba[~mask,:],multi_class='ovo'))
            else:
                print(f'predicted_{allprops[i]}, mean roc auc: ', roc_auc_score(y_true[~mask]==1, y_pred_proba[~mask,1]))
            print(classification_report(y_true[~mask], y_pred[~mask],labels=[ index for index in range(y_pred_proba.shape[1])], target_names= list(self.labelnames[allprops[i]].values())))
            i+=1
            
 
    ###########################################
    def print_metrics(self):
        """    
        prints the model and results
        
        Returns: 
            list of strings of the printed models
        """
        if self.merged_model:
            output_str='Merged model, print statement disabled'
            print(output_str)
            return output_str
        else:
            CEND      = '\33[0m'
            CGREEN  = '\33[32m'
            CYELLOW = '\33[33m'
            CBOLD     = '\33[1m'
            output_str=[]
            for p in self.models:
                if isinstance(self.models[p], VotingClassifier):
                    models='\n'.join([str(m[1].steps[1][1])+':'+str(m[1].steps[2][1]) if m[1].steps[1][1]!='passthrough' else str(m[1].steps[2][1]) for m in self.models[p].estimators]  )
                else:
                    models='\n'.join([str(self.models[p].steps[1][1])+': '+str(self.models[p].steps[2][1]) if self.models[p].steps[1][1]!='passthrough' else str(self.models[p].steps[2][1])]  )
                m=self.metrics[p]
                print(CBOLD+p+CEND ,':', m)
                print('List of models:')
                print(models)
                output_str.append(models)
            return output_str
        

###########################################################



class FeatureGenerationSingleModelClassifier(FeatureGenerationClassifier):
    """
    FeatureGenerationSingleModelClassifier is a specialization of FeatureGenerationClassifier
    that finds a single classifier
    """
    
    def create_modfinder(self):
        """    
        Returns the appropriate derivation of ModelFinder
        
        Returns: 
            ClassificationFinder(verbose,NestedCVSingleModelSearch,outer_jobs)
        """
        return ClassificationFinder(self.verbose,NestedCVSingleModelSearch(self.verbose),self.outer_jobs)


class FeatureGenerationStackingClassifiers(FeatureGenerationClassifier):
    """
    FeatureGenerationStackingClassifiers is a specialization of FeatureGenerationClassifier
    that finds a list of stacking classifiers
    """
    
    def create_modfinder(self):
        """    
        Returns the appropriate derivation of ModelFinder
        
        Returns: 
            ClassificationFinder(verbose,NestedCVBaseStackingSearch,outer_jobs)
        """
        return ClassificationFinder(self.verbose,NestedCVBaseStackingSearch(self.verbose),self.outer_jobs)
    
    def print_metrics(self):
        """    
        prints the model and results
        
        Returns: 
            list of strings of the printed models
        """
        if self.merged_model:
            output_str='Merged model, print statement disabled'
            print(output_str)
            return output_str
        else:
            CEND      = '\33[0m'
            CGREEN  = '\33[32m'
            CYELLOW = '\33[33m'
            CBOLD     = '\33[1m'
            output_str=[]
            for p in self.models:
                models=''
                for index,stacked_estimator_tuple in enumerate(self.models[p].estimators):
                    stacked_estimator=stacked_estimator_tuple[1]
                    models+=CBOLD+'Stacked model '+str(index+1)+": "+CEND+'Base estimators: '+ ','.join(['\n'.join([str(m[1].steps[1][1])+':'+str(m[1].steps[2][1]) if m[1].steps[1][1]!='passthrough' else str(m[1].steps[2][1])]) for m in stacked_estimator.estimators])+'\n'+CBOLD+'Stacking final estimator: '+CEND+':'.join([str(s) for s in [stacked_estimator.final_estimator.steps[0][1],stacked_estimator.final_estimator.steps[1][1], stacked_estimator.final_estimator.steps[2][1]]])+'\n'
                m=self.metrics[p]
                print(CBOLD+p+CEND ,':', m)
                print('List of models:')
                print(models)
                output_str.append(models)
            return output_str


class FeatureGenerationTopstackingStackingclassifier(FeatureGenerationClassifier):
    """
    FeatureGenerationTopstackingStackingclassifier is a specialization of FeatureGenerationClassifier
    that finds a list of stacking classifiers with their output given to a toplevel stacking classifier
    """
    
    def create_modfinder(self):
        """    
        Returns the appropriate derivation of ModelFinder
        
        Returns: 
            ClassificationFinder(verbose,NestedCVBaseStackingSearch(top_stacking=True),outer_jobs)
        """
        return ClassificationFinder(self.verbose,NestedCVBaseStackingSearch(self.verbose,top_stacking=True),self.outer_jobs)
    
    def print_metrics(self):
        """    
        prints the model and results
        
        Returns: 
            list of strings of the printed models
        """
        if self.merged_model:
            output_str='Merged model, print statement disabled'
            print(output_str)
            return output_str
        else:
            CEND      = '\33[0m'
            CGREEN  = '\33[32m'
            CYELLOW = '\33[33m'
            CBOLD     = '\33[1m'
            output_str=[]
            for p in self.models:
                top_stacked_estimator=self.models[p]
                models=''
                for index,stacked_estimator_tuple in enumerate(top_stacked_estimator.estimators):
                    stacked_estimator=stacked_estimator_tuple[1]
                    models+=CBOLD+'Stacked model '+str(index+1)+": "+CEND+'Base estimators: '+ ','.join(['\n'.join([str(m[1].steps[1][1])+':'+str(m[1].steps[2][1]) if m[1].steps[1][1]!='passthrough' else str(m[1].steps[2][1])]) for m in stacked_estimator.estimators])+'\n'+CBOLD+'Stacking final estimator: '+CEND+':'.join([str(s) for s in [stacked_estimator.final_estimator.steps[0][1],stacked_estimator.final_estimator.steps[1][1], stacked_estimator.final_estimator.steps[2][1]]])+'\n'
                models+=CBOLD+'Stacking global final estimator: '+CEND+':'.join([str(s) for s in [top_stacked_estimator.final_estimator.steps[0][1],top_stacked_estimator.final_estimator.steps[1][1], top_stacked_estimator.final_estimator.steps[2][1]]])
                m=self.metrics[p]
                print(CBOLD+p+CEND ,':', m)
                print('List of models:')
                print(models)
                output_str.append(models)
            return output_str


class FeatureGenerationTopstackingClassifier(FeatureGenerationClassifier):
    """
    FeatureGenerationTopstackingClassifier is a specialization of FeatureGenerationClassifier
    that finds a list of base level methods and these are used as estimators for a top level stacking model
    """
    
    def create_modfinder(self):
        """    
        Returns the appropriate derivation of ModelFinder
        
        Returns: 
            ClassificationFinder(verbose,NestedCVModelSearch(top_stacking=True),outer_jobs)
        """
        return ClassificationFinder(self.verbose,NestedCVModelSearch(self.verbose,top_stacking=True),outer_jobs=self.outer_jobs)
    
    def print_metrics(self):
        """    
        prints the model and results
        
        Returns: 
            list of strings of the printed models
        """
        if self.merged_model:
            output_str='Merged model, print statement disabled'
            print(output_str)
            return output_str
        else:
            CEND      = '\33[0m'
            CGREEN  = '\33[32m'
            CYELLOW = '\33[33m'
            CBOLD     = '\33[1m'
            output_str=[]
            for p in self.models:
                stacked_estimator=self.models[p]
                models=CBOLD+'Base estimators: '+CEND+ ','.join(['\n'.join([str(m[1].steps[1][1])+':'+str(m[1].steps[2][1]) if m[1].steps[1][1]!='passthrough' else str(m[1].steps[2][1])]) for m in stacked_estimator.estimators])+'\n'+CBOLD+'Stacking final estimator: '+CEND+':'.join([str(s) for s in [stacked_estimator.final_estimator.steps[0][1],stacked_estimator.final_estimator.steps[1][1], stacked_estimator.final_estimator.steps[2][1]]])
                m=self.metrics[p]
                print(CBOLD+p+CEND ,':', m)
                print('List of models:')
                print(models)
                output_str.append(models)
            return output_str


class FeatureGenerationSingleStackClassifier(FeatureGenerationClassifier):
    """
    FeatureGenerationSingleStackClassifier is a specialization of FeatureGenerationClassifier
    that finds a single stacking model
    """
    
    def create_modfinder(self):
        """    
        Returns the appropriate derivation of ModelFinder
        
        Returns: 
            ClassificationFinder(verbose,NestedCVSingleStackSearch,outer_jobs)
        """
        return ClassificationFinder(self.verbose,NestedCVSingleStackSearch(self.verbose),outer_jobs=self.outer_jobs)
    
    def print_metrics(self):
        """    
        prints the model and results
        
        Returns: list of strings of the printed models
        """
        if self.merged_model:
            output_str='Merged model, print statement disabled'
            print(output_str)
            return output_str
        else:
            CEND      = '\33[0m'
            CGREEN  = '\33[32m'
            CYELLOW = '\33[33m'
            CBOLD     = '\33[1m'
            output_str=[]
            for p in self.models:
                stacked_estimator=self.models[p]
                models=CBOLD+'Base estimators: '+CEND+ ','.join(['\n'.join([str(m[1].steps[1][1])+':'+str(m[1].steps[2][1]) if m[1].steps[1][1]!='passthrough' else str(m[1].steps[2][1])]) for m in stacked_estimator.estimators])+'\n'+CBOLD+'Stacking final estimator: '+CEND+':'.join([str(s) for s in [stacked_estimator.final_estimator.steps[0][1],stacked_estimator.final_estimator.steps[1][1], stacked_estimator.final_estimator.steps[2][1]]])
                m=self.metrics[p]
                print(CBOLD+p+CEND ,':', m)
                print('List of models:')
                print(models)
                output_str.append(models)
            return output_str


########################################################### 

class FeatureGenerationSingleModelRegressor(FeatureGenerationRegressor):
    """
    FeatureGenerationSingleModelRegressor is a specialization of FeatureGenerationRegressor
    that finds a single regressor
    """
    
    def create_modfinder(self):
        """    
        Returns the appropriate derivation of ModelFinder
        
        Returns: 
            RegressionFinder(verbose,NestedCVSingleModelSearch,outer_jobs)
        """
        return RegressionFinder(self.verbose,NestedCVSingleModelSearch(self.verbose),outer_jobs=self.outer_jobs)

class FeatureGenerationStackingRegressors(FeatureGenerationRegressor):
    """
    FeatureGenerationStackingRegressors is a specialization of FeatureGenerationRegressor
    that finds a list of stacking regressors
    """
    
    def create_modfinder(self):
        """    
        Returns the appropriate derivation of ModelFinder
        
        Returns: 
            RegressionFinder(verbose,NestedCVBaseStackingSearch,outer_jobs)
        """
        return RegressionFinder(self.verbose,NestedCVBaseStackingSearch(self.verbose),outer_jobs=self.outer_jobs)
    
    def print_metrics(self):
        """    
        prints the model and results
        
        Returns: 
            list of strings of the printed models
        """
        if self.merged_model:
            output_str='Merged model, print statement disabled'
            print(output_str)
            return output_str
        else:
            CEND      = '\33[0m'
            CGREEN  = '\33[32m'
            CYELLOW = '\33[33m'
            CBOLD     = '\33[1m'
            output_str=[]
            for p in self.models:
                models=''
                for index,stacked_estimator in enumerate(self.models[p]):
                    models+=CBOLD+'Stacked model '+str(index+1)+": "+CEND+'Base estimators: '+ ','.join(['\n'.join([str(m[1].steps[1][1])+':'+str(m[1].steps[2][1]) if m[1].steps[1][1]!='passthrough' else str(m[1].steps[2][1])]) for m in stacked_estimator.estimators])+'\n'+CBOLD+'Stacking final estimator: '+CEND+':'.join([str(s) for s in [stacked_estimator.final_estimator.steps[0][1],stacked_estimator.final_estimator.steps[1][1], stacked_estimator.final_estimator.steps[2][1]]])+'\n'
                m=self.metrics[p]
                print(CBOLD+p+CEND ,':', m)
                print('List of models:')
                print(models)
                output_str.append(models)
            return output_str


class FeatureGenerationSingleStackRegressor(FeatureGenerationRegressor):
    """
    FeatureGenerationSingleStackRegressor is a specialization of FeatureGenerationRegressor
    that finds a single stacking regressor
    """
    
    def create_modfinder(self):
        """    
        Returns the appropriate derivation of ModelFinder
        
        Returns: 
            RegressionFinder(verbose,NestedCVSingleStackSearch,outer_jobs)
        """
        return RegressionFinder(self.verbose,NestedCVSingleStackSearch(self.verbose),outer_jobs=self.outer_jobs)
    
    def print_metrics(self):
        """    
        prints the model and results
        
        Returns: 
            list of strings of the printed models
        """
        if self.merged_model:
            output_str='Merged model, print statement disabled'
            print(output_str)
            return output_str
        else:
            CEND      = '\33[0m'
            CGREEN  = '\33[32m'
            CYELLOW = '\33[33m'
            CBOLD     = '\33[1m'
            output_str=[]
            for p in self.models:
                stacked_estimator=self.models[p]
                models=CBOLD+'Base estimators: '+CEND+ ','.join(['\n'.join([str(m[1].steps[1][1])+':'+str(m[1].steps[2][1]) if m[1].steps[1][1]!='passthrough' else str(m[1].steps[2][1])]) for m in stacked_estimator.estimators])+'\n'+CBOLD+'Stacking final estimator: '+CEND+':'.join([str(s) for s in [stacked_estimator.final_estimator.steps[0][1],stacked_estimator.final_estimator.steps[1][1], stacked_estimator.final_estimator.steps[2][1]]])
                m=self.metrics[p]
                print(CBOLD+p+CEND ,':', m)
                print('List of models:')
                print(models)
                output_str.append(models)
            return output_str

class FeatureGenerationTopStackingRegressor(FeatureGenerationRegressor):
    """
    FeatureGenerationTopStackingRegressor is a specialization of FeatureGenerationRegressor
    that builts a stacking regressors on base level estimators found during the inner folds 
    """
    
    def create_modfinder(self):
        """    
        Returns the appropriate derivation of ModelFinder
        
        Returns: 
            RegressionFinder(verbose,NestedCVModelSearch(top_stacking=True),outer_jobs)
        """
        return RegressionFinder(self.verbose,NestedCVModelSearch(self.verbose,top_stacking=True),outer_jobs=self.outer_jobs)
    
    def print_metrics(self):
        """    
        prints the model and results
        
        Returns: 
            list of strings of the printed models
        """
        if self.merged_model:
            output_str='Merged model, print statement disabled'
            print(output_str)
            return output_str
        else:
            CEND      = '\33[0m'
            CGREEN  = '\33[32m'
            CYELLOW = '\33[33m'
            CBOLD     = '\33[1m'
            output_str=[]
            for p in self.models:
                stacked_estimator=self.models[p]
                models=CBOLD+'Base estimators: '+CEND+ ','.join(['\n'.join([str(m[1].steps[1][1])+':'+str(m[1].steps[2][1]) if m[1].steps[1][1]!='passthrough' else str(m[1].steps[2][1])]) for m in stacked_estimator.estimators])+'\n'+CBOLD+'Stacking final estimator: '+CEND+':'.join([str(s) for s in [stacked_estimator.final_estimator.steps[0][1],stacked_estimator.final_estimator.steps[1][1], stacked_estimator.final_estimator.steps[2][1]]])
                m=self.metrics[p]
                print(CBOLD+p+CEND ,':', m)
                print('List of models:')
                print(models)
                output_str.append(models)
            return output_str
        
        
#################################################################
class FeatureGenerationSingleModelRegressorClassifier(FeatureGenerationRegressionClassifier):
    """
    FeatureGenerationSingleModelRegressor is a specialization of FeatureGenerationRegressor
    that finds a single regressor
    """
    
    def create_modfinder(self):
        """    
        Returns the appropriate derivation of ModelFinder
        
        Returns: 
            RegressionFinder(verbose,NestedCVSingleModelSearch,outer_jobs)
        """
        return RegressionFinder(self.verbose,NestedCVSingleModelSearch(self.verbose),outer_jobs=self.outer_jobs)

class FeatureGenerationStackingRegressorsClassifier(FeatureGenerationRegressionClassifier):
    """
    FeatureGenerationStackingRegressors is a specialization of FeatureGenerationRegressor
    that finds a list of stacking regressors
    """
    
    def create_modfinder(self):
        """    
        Returns the appropriate derivation of ModelFinder
        
        Returns: 
            RegressionFinder(verbose,NestedCVBaseStackingSearch,outer_jobs)
        """
        return RegressionFinder(self.verbose,NestedCVBaseStackingSearch(self.verbose),outer_jobs=self.outer_jobs)
    
    def print_metrics(self):
        """    
        prints the model and results
        
        Returns: 
            list of strings of the printed models
        """
        if self.merged_model:
            output_str='Merged model, print statement disabled'
            print(output_str)
            return output_str
        else:
            CEND      = '\33[0m'
            CGREEN  = '\33[32m'
            CYELLOW = '\33[33m'
            CBOLD     = '\33[1m'
            output_str=[]
            for p in self.models:
                models=''
                for index,stacked_estimator in enumerate(self.models[p]):
                    models+=CBOLD+'Stacked model '+str(index+1)+": "+CEND+'Base estimators: '+ ','.join(['\n'.join([str(m[1].steps[1][1])+':'+str(m[1].steps[2][1]) if m[1].steps[1][1]!='passthrough' else str(m[1].steps[2][1])]) for m in stacked_estimator.estimators])+'\n'+CBOLD+'Stacking final estimator: '+CEND+':'.join([str(s) for s in [stacked_estimator.final_estimator.steps[0][1],stacked_estimator.final_estimator.steps[1][1], stacked_estimator.final_estimator.steps[2][1]]])+'\n'
                m=self.metrics[p]
                print(CBOLD+p+CEND ,':', m)
                print('List of models:')
                print(models)
                output_str.append(models)
            return output_str


class FeatureGenerationSingleStackRegressorClassifier(FeatureGenerationRegressionClassifier):
    """
    FeatureGenerationSingleStackRegressor is a specialization of FeatureGenerationRegressor
    that finds a single stacking regressor
    """
    
    def create_modfinder(self):
        """    
        Returns the appropriate derivation of ModelFinder
        
        Returns: 
            RegressionFinder(verbose,NestedCVSingleStackSearch,outer_jobs)
        """
        return RegressionFinder(self.verbose,NestedCVSingleStackSearch(self.verbose),outer_jobs=self.outer_jobs)
    
    def print_metrics(self):
        """    
        prints the model and results
        
        Returns: 
            list of strings of the printed models
        """
        if self.merged_model:
            output_str='Merged model, print statement disabled'
            print(output_str)
            return output_str
        else:
            CEND      = '\33[0m'
            CGREEN  = '\33[32m'
            CYELLOW = '\33[33m'
            CBOLD     = '\33[1m'
            output_str=[]
            for p in self.models:
                stacked_estimator=self.models[p]
                models=CBOLD+'Base estimators: '+CEND+ ','.join(['\n'.join([str(m[1].steps[1][1])+':'+str(m[1].steps[2][1]) if m[1].steps[1][1]!='passthrough' else str(m[1].steps[2][1])]) for m in stacked_estimator.estimators])+'\n'+CBOLD+'Stacking final estimator: '+CEND+':'.join([str(s) for s in [stacked_estimator.final_estimator.steps[0][1],stacked_estimator.final_estimator.steps[1][1], stacked_estimator.final_estimator.steps[2][1]]])
                m=self.metrics[p]
                print(CBOLD+p+CEND ,':', m)
                print('List of models:')
                print(models)
                output_str.append(models)
            return output_str

class FeatureGenerationTopStackingRegressorClassifier(FeatureGenerationRegressionClassifier):
    """
    FeatureGenerationTopStackingRegressor is a specialization of FeatureGenerationRegressor
    that builts a stacking regressors on base level estimators found during the inner folds 
    """
    
    def create_modfinder(self):
        """    
        Returns the appropriate derivation of ModelFinder
        
        Returns: 
            RegressionFinder(verbose,NestedCVModelSearch(top_stacking=True),outer_jobs)
        """
        return RegressionFinder(self.verbose,NestedCVModelSearch(self.verbose,top_stacking=True),outer_jobs=self.outer_jobs)
    
    def print_metrics(self):
        """    
        prints the model and results
        
        Returns: 
            list of strings of the printed models
        """
        if self.merged_model:
            output_str='Merged model, print statement disabled'
            print(output_str)
            return output_str
        else:
            CEND      = '\33[0m'
            CGREEN  = '\33[32m'
            CYELLOW = '\33[33m'
            CBOLD     = '\33[1m'
            output_str=[]
            for p in self.models:
                stacked_estimator=self.models[p]
                models=CBOLD+'Base estimators: '+CEND+ ','.join(['\n'.join([str(m[1].steps[1][1])+':'+str(m[1].steps[2][1]) if m[1].steps[1][1]!='passthrough' else str(m[1].steps[2][1])]) for m in stacked_estimator.estimators])+'\n'+CBOLD+'Stacking final estimator: '+CEND+':'.join([str(s) for s in [stacked_estimator.final_estimator.steps[0][1],stacked_estimator.final_estimator.steps[1][1], stacked_estimator.final_estimator.steps[2][1]]])
                m=self.metrics[p]
                print(CBOLD+p+CEND ,':', m)
                print('List of models:')
                print(models)
                output_str.append(models)
            return output_str

#@staticmethod
def load_model(model_file=None,use_gpu=True,rtol=1e-6, atol=1e-8,relative_error=True,retrieve_reproducability_errors=False
              , result_dict:dict=None
              , metadata:dict=None) :
    """    
    loads a stacking model and evaluate the reproducable output
    
    Args:
         model_file: .pt file with the model
         use_gpu: use_gpu
         rtol: relative tolerance for errors
         atol: absolute tolerance for errors
         relative_error: compute relative error of reproducable errors
         retrieve_reproducability_errors: return dictionary of with errors
    
    Returns: 
        model or tuple of model and dictionary of errors
    """
    print(f"Loading model from file {model_file}...")

    aa=torch.load(model_file , map_location='cpu',weights_only=False)
    model=aa['model']
    if result_dict is not None:
        for key, value in aa['res_dict'].items():
            result_dict[key]=value 
    if metadata is not None:
        metadata['model_metadata']=aa['metadata'] 
    model.load_state_dict(aa['model_state_dict'])

    device =torch.device('cpu')
    if use_gpu and  torch.cuda.is_available(): 
        device = torch.device(f'cuda:0')
    print('using device:',device)
    model=model.to(device)

    reproducability_errors={}
    try:
        reproducability_errors=model.test_reproducable_output(rtol=rtol, atol=atol,verbose=True,relative_error=relative_error) 
    except AttributeError:
        pass

    if retrieve_reproducability_errors:
        return model,reproducability_errors
    return model    


    

def save_model(model, f_path,create_reproducability_output=True, result_dict:dict=None):
    """    
    save the model and generates the reproducable output
    
    Args:
         model: the stacking model
         f_path: the .pt file where the model should be saved
         create_reproducability_output: generate reproducability error
    """
    #check if reproducability generation functionality is available within the model
    if hasattr(model,'reproducable_output'):
        if create_reproducability_output:
            model.generate_reproducable_output()     
    print(f"Saving config file to file {f_path}...")
    out={}
    if hasattr(model, 'clean_features'):model.clean_features()
    out['model']= model
    out['model_state_dict']= model.state_dict()
    if result_dict is not None: 
        out['res_dict']=result_dict
    torch.save( out,f_path)
        


import numpy as np, pandas as pd
from  matplotlib import pyplot as plt
from automol.property_prep import add_stereo_smiles
from automol.property_prep import make_category
from automol.validation import stratified_validation, leave_grp_out_validation, mixed_validation
from automol.property_prep import add_stereo_smiles,validate_rdkit_smiles, add_rdkit_standardized_smiles
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.base import BaseEstimator, ClassifierMixin,RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from automol.stacking_util import get_clustering_algorithm


"""! @brief  implementation of the multi-layer perceptron wrappers.

@section author_section Author(s)

Authors: Mazen Ahmad, Joris Tavernier, Natalia Dyubankova, Marvin Steijaert

Contact: mahmad13@its.jnj.com, jtaverni@its.jnj.com

All rights reserved, Janssen Pharmaceutica and Open Analytics NV, 2021-2022. 
"""
##
# @file mlpwrappers.py
#
# @brief  implementation of the multi-layer perceptron wrappers.
#
# @section libraries_sensors Libraries/Modules
# - sklearn.utils.validation
# - sklearn.base 
# - sklearn.neural_network
# - sklearn.utils.multiclass
#
# @section author_sensors Author(s)
#
#Authors: Mazen Ahmad, Joris Tavernier, Natalia Dyubankova, Marvin Steijaert
#
#All rights reserved, Janssen Pharmaceutica and Open Analytics NV, 2021-2022. 

class HammingMultioutputScore():   
    def __call__(self,y_true,y_pred):
        return -1*np.sum(np.not_equal(y_true, y_pred))/float(y_true.size)

    def __name__(self):
        return 'HammingMultioutputScore'


class EmptyClassifier(BaseEstimator, ClassifierMixin):       
    def fit(self, X, y,**fit_params):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)        
        return self
    
    def predict(self, X):
        X = check_array(X)
        return np.repeat(0, X.shape[0])
    
    def predict_proba(self, X):
        X = check_array(X)
        return np.random.rand(X.shape[0],self.classes_)
    

class EmptyRegressor(BaseEstimator, RegressorMixin):       
    def fit(self, X, y,**fit_params):
        X, y = check_X_y(X, y)
        return self
    
    def predict(self, X):
        X = check_array(X)
        return np.random.randn(X.shape[0])
    

    

from automol.stacking_methodarchive import ClassifierArchive,ReducedimArchive,RegressorArchive
from automol.feature_generators import FeatureGenerator


class RandomNanInjector(FeatureGenerator):
    def __init__(self,feature_generator):
        self.__feature_generator=feature_generator
        self.__seed=42
        
    def get_nb_features(self):
        """!
        getter for the number of features.
        
        includes an assert that number of features is positive.
        
        @return number of features
        """
        return self.__feature_generator.get_nb_features()
    
    def check_consistency(self):
        """!
        checks if the number of features is positive and the length of the feature names equal the number of features
        """
        
        return self.__feature_generator.check_consistency()
    
    
    def get_names(self):
        """!
        getter for the names of the features
        
        @return list of names
        """
        return self.__feature_generator.get_names()
    
    def get_generator_name(self):
        """!
        getter for the generator name
        
        @return generator name
        """
        return self.__feature_generator.get_generator_name()
        
    def set_seed(self, seed:int=3):
        self.__seed=seed
        
    def generate(self, smiles):
        np.random.seed(self.__seed)
        X=self.__feature_generator.generate(smiles)
        for index in np.random.randint(0,len(smiles), size=3):
            X[index,:]=np.array(self.__feature_generator.get_nb_features()*[np.nan])
        return X

    
class test_util:
    def __init__(self):
        pass

    
    def get_method_archive_with_empty_estimators(self,task,method_prefix, distribution_defaults,hyperopt_defaults, random_state, xgb_threads, rfr_threads, method_jobs):
        if task=='clf':
            method_archive=ClassifierArchive(method_prefix=method_prefix, distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults, random_state=random_state, xgb_threads=xgb_threads, rfr_threads=rfr_threads, method_jobs=method_jobs)
            method_archive.add_method('empty',EmptyClassifier(),{})
            return method_archive
        else:
            method_archive=RegressorArchive(method_prefix=method_prefix, distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults, random_state=random_state, xgb_threads=xgb_threads, rfr_threads=rfr_threads, method_jobs=method_jobs)
            method_archive.add_method('empty',EmptyRegressor(),{})
            return method_archive
    
        
        

    def read_data(self,file_name,smiles_column,verbose=0,nb_samples=500,standard_smiles_column='stereo_SMILES',adme_il17=False,check_rdkit_desc=False):
        df= pd.read_csv(file_name, na_values = ['NAN', '?','NaN'])
        if adme_il17:
            add_rdkit_standardized_smiles(df, smiles_column,verbose=False,outname=standard_smiles_column)
        else:
            add_stereo_smiles(df,smiles_column,verbose=verbose,outname=standard_smiles_column)
        if check_rdkit_desc: 
            validate_rdkit_smiles(df, standard_smiles_column,verbose=verbose)

        add_stereo_smiles(df,smiles_column,verbose=verbose)
        df.dropna(inplace=True, subset = [standard_smiles_column])
        df=df.iloc[1:nb_samples,:]
        return df
    
    def create_clf_validation(self,df,properties,class_properties,strategy,categorical,stacked_model,standard_smiles_column,df_smiles,
                             test_size,val_clustering, val_km_groups,val_butina_cutoff, val_include_chirality,verbose, random_state,minority_nb=5,clustering_algorithm=None, chem_clustering_algorithm=None):
        
        if clustering_algorithm is None and np.random.rand(1,1)<0.5:
            clustering_algorithm=get_clustering_algorithm(clustering=val_clustering,
                                 n_clusters=val_km_groups,
                                 cutoff=val_butina_cutoff,
                                 include_chirality=val_include_chirality,
                                 verbose=verbose,
                                 random_state=random_state,
                                 feature_generators=stacked_model.feature_generators,
                                 used_features='Bottleneck')
        
        leave_grp_out=None
        prop_cliff_dict=None
        if strategy=='mixed':
            prop_cliff_butina_th=0.6
            rel_prop_cliff=0.3
            mix_coef_dict={ 'prop_cliffs': 0.3,'leave_group_out': 0.3 ,'stratified': 0.4}
            if categorical:
                prop_cliff_butina_th=0.5
                print('Activity cliffs with categorical properties: similar compounds with different class are considered as an activity cliff ')
            Train, Validation,leave_grp_out, prop_cliff_dict = mixed_validation(df_orig=df,properties=properties,stacked_model=stacked_model,standard_smiles_column=standard_smiles_column,
                                                            prop_cliff_cut=rel_prop_cliff,prop_cliff_butina=prop_cliff_butina_th,test_size=test_size,clustering=val_clustering,
                                                          n_clusters=val_km_groups,cutoff=val_butina_cutoff,include_chirality=val_include_chirality,
                                                          verbose=verbose,random_state=random_state,mix_dict=mix_coef_dict, categorical_data=categorical,
                                                           minority_nb=minority_nb,clustering_algorithm=clustering_algorithm, chem_clustering_algorithm=chem_clustering_algorithm)
        elif strategy=='stratified':
            Train, Validation = stratified_validation(df,class_properties,stacked_model,df_smiles,test_size=test_size,clustering=val_clustering,
                                                          n_clusters=val_km_groups,cutoff=val_butina_cutoff,include_chirality=val_include_chirality,
                                                          verbose=verbose,random_state=random_state, minority_nb=minority_nb,clustering_algorithm=clustering_algorithm)
        else:
            Train, Validation = leave_grp_out_validation(df,class_properties,stacked_model,df_smiles,test_size=test_size,clustering=val_clustering,
                                                          n_clusters=val_km_groups,cutoff=val_butina_cutoff,include_chirality=val_include_chirality,
                                                          verbose=verbose,random_state=random_state,clustering_algorithm=clustering_algorithm)
            leave_grp_out=np.arange(len(Validation))
            
        return Train,Validation,leave_grp_out,prop_cliff_dict

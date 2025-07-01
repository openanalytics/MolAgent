"""implementation of the different feature generators and their base class.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""

import time
import numpy as np 


def get_nb_feature_multiplier(relative=False,feature_op='concat'):
    '''helper function to retrieve number of feature multiplier

    Args:
        relative boolean: to indicate paired ligand modeling or single, true for paired
        feature_op: string to operation of paired ligand features
    '''
    if relative:
        if feature_op=='plus' or feature_op=='add' or feature_op=='addition':
            return 1
        elif feature_op=='subtract' or feature_op=='subtraction':
            return 1
        else:
            return 2
    else:
        return 1

class SingleLigand():
    def __init__(self,verbose=False):
        self.verbose=verbose
        
    def precompute_features(self,smiles=None,gen_features={},feature_generators={},feature_list=None, original_indices=None, indices=None):
        """    
        generate all features
        
        Args:
             smiles: column name of the smiles
             df: dataframe
             gen_features: dictionary of already computed features
             feature_generators: dictionary with feature generators
             feature_list: list of feature generation keys
             original_indices: unused in case of single ligand
             indices: unused in case of single ligand
        """
        start_precompute=time.time()

        if feature_list is not None:
            for feat_key in feature_list:
                assert feat_key in feature_generators, f'provided feature generator key not present in dictionary of feature_generators, available keys: {feature_generators.keys()}'
                if feat_key not in gen_features:
                    gen_features[feat_key]=feature_generators[feat_key].generate(smiles)
                    gen_features[f'{feat_key}_names']=feature_generators[feat_key].get_names()

        if self.verbose:
            minutes, seconds = divmod(time.time() - start_precompute, 60)
            print(f'performed feature compute in {int(minutes)} m and {int(seconds)} s' )
            
        return gen_features
    
    def apply_feature_operation(self,xlist):
           
        return np.concatenate( xlist, axis=-1)

            
    def create_X_y(self,df,prop, smiles,feature_dict,features,multiprop,indices=None):
        """    
        generate all features
        
        Args:
             df: dataframe
             prop:
             smiles:
             feature_dict:
             features: list of feature keys
             multiprop: unused in case of single ligand
             indices: unused in case of single ligand
        """
        assert prop in df 
        na=df[prop].isna()
        feature_na=np.zeros(len(na), dtype=bool)
        for i, feature_name in enumerate(features):
            feature_na=np.logical_or([ np.isnan(row).any() for j, row in enumerate(feature_dict[feature_name])],feature_na)
        na=np.logical_or(na,feature_na)
        
        if self.verbose:
            print(f'Deleted the following smiles due to nan features/property value:{list(df[smiles][na])}')
        X=np.concatenate( [feature_dict[k][~na] for k in features], axis=-1)
        y=df.loc[~na,prop].values
        
        if multiprop:
            y=np.concatenate([[v] for v in y], axis=0)
        
        return X,y,na
    
    def update_groups(self,groups,indices):
        return groups
    
    def get_pairs(self,df,pairs_col:str='pairs'):
        return None,None
    
    
    
class PairedLigands():
    def __init__(self,verbose=False,feature_operation:str='concat',property_operation:str='minus'):
        self.verbose=verbose
        self.feature_operation=feature_operation
        self.property_operation=property_operation
        
    def _all_to_all(self,n:int=0):
        l=[]
        for i in range(n):
            for j in range(i+1,n):
                l.append((i,j))
        return l
        
    def precompute_features(self,smiles=None,gen_features={},feature_generators={},feature_list=None,original_indices=None,indices=None):
        """    
        generate all features

        We pass both the original indices as the new indices as pairs. The original in case there is a predefined dataset
        containing both training and validation data (expensive preprocessed molecular graph dataset for example). The original
        indices refer to the index in this dataset. The indices are the new indices and refer directly to the training set 
        
        Args:
             smiles: column name of the smiles
             df: dataframe
             gen_features: dictionary with already generated features
             feature_generators: dictionary with feature generators
             feature_list: list of used feature generation keys
             original_indices: the original pairs of indices, corresponding to a preprocessed dataset before data splitting 
             indices: indices for splitted datasets 
        """
        start_precompute=time.time()
        features={}
        
        if indices is None:
            indices=self._all_to_all(len(smiles))
        if original_indices is None:
            original_indices=indices
            
        number_pairs=len(indices)

        if feature_list is not None:
            for feat_key in feature_list:
                assert feat_key in feature_generators, f'provided feature generator key not present in dictionary of feature_generators, available keys: {feature_generators.keys()}'
                if feat_key not in gen_features:
                    gen_features[feat_key]=feature_generators[feat_key].generate_w_pairs(smiles,original_indices=original_indices,new_indices=indices)
                    gen_features[f'{feat_key}_names']=feature_generators[feat_key].get_names()
                #check right size of data
                elif gen_features[feat_key].shape[0]<number_pairs:
                    gen_features[feat_key]=feature_generators[feat_key].generate_w_pairs(smiles,original_indices=original_indices,new_indices=indices)
                    gen_features[f'{feat_key}_names']=feature_generators[feat_key].get_names()
                    

        if self.verbose:
            minutes, seconds = divmod(time.time() - start_precompute, 60)
            print(f'performed feature compute in {int(minutes)} m and {int(seconds)} s' )
            
        return gen_features
            
    def update_groups(self,groups,indices):
        """Defines cluster groups for pairs based on groups of single ligands. The smallest group is chosen
         in case none of the pair if ligands is in a minority group.

         args:
            groups: cluster indices for single ligands
            indices: pair defining tupels of indices
        """
        grps=[]
        for (i,j) in indices:
            if groups[i]=='minorities' and groups[j] =='minorities':
                grps.append('minorities')
            elif groups[i]=='minorities':
                grps.append(j)
            elif groups[j]=='minorities':
                grps.append(i)
            else:
                grps.append(min(i,j))
        return np.array(grps)
    
    def apply_property_operation(self,y1,y2):
        ''' Applies predefined operation on the property values

        Args:
            y1: property value of ligand one
            y2: property value of ligand two
        '''
        if self.property_operation=='plus' or self.property_operation=='add' or self.property_operation=='addition':
            return y1+y2
        elif self.property_operation=='times' or self.property_operation=='multiply':
            return y1*y2
        elif self.property_operation=='divide' or self.property_operation=='division':
            assert y2>0
            return y1/y2
        elif self.property_operation=='identical' or self.property_operation=='equality'or self.property_operation=='==':
            if y1==y2:
                return 1
            else:
                return 0
        else:
            return y1-y2
        
    def apply_feature_operation(self,xlist):
        '''apply feature operation to list of features. 

        The paired feature matrices contain both ligand features concatenated. Here we add/subtract or simply leave as is. 

        Args:
            xlist: list of feature matrices for each feature type
        '''
        x_n=[]
        if self.feature_operation=='plus' or self.feature_operation=='add' or self.feature_operation=='addition':
            for Xi in xlist:
                half=Xi.shape[1]/2
                x_n.append(Xi[:,:half]+Xi[:,half:])
        elif self.feature_operation=='subtract' or self.feature_operation=='subtraction':
            for Xi in xlist:
                half=Xi.shape[1]/2
                x_n.append(Xi[:,:half]-Xi[:,half:])
        else:
            x_n=xlist
            
        return np.concatenate( x_n, axis=-1)
        
    def create_X_y(self,df,prop, smiles,feature_dict,features,multiprop,indices):
        '''Creates the features matrix and target

        Args:
            df: Dataframe
            prop: used property name (column in df) 
            smiles: column name containing the smiles
            feature_dict: dictionary with generated features
            features: list of used features
            multiprop: boolean indicating multiple properties
            indices: paired indices
        '''
        assert prop in df 
        if indices is None:
            indices=self._all_to_all(len(df[smiles]))
        
        feature_na=np.zeros(len(indices), dtype=bool)
        for i, feature_name in enumerate(features):
            feature_na=np.logical_or([ np.isnan(row).any() for j, row in enumerate(feature_dict[feature_name])],feature_na)
        
        na=df[prop].isna().values
        y=df[prop].values
        y_p=np.zeros(len(indices))
        na_p=np.zeros(len(indices))  
        for idx,(i,j) in enumerate(indices):
            if len(y_p.shape)>1:
                y_p[idx,:]=self.apply_property_operation(y[i,:],y[j,:])
            else:
                y_p[idx]=self.apply_property_operation(y[i],y[j])
            #print(i,j,na.shape,na_p.shape)
            na_p[idx]=na[i]+na[j]
            
        na=np.logical_or(na_p>0,feature_na)
        if self.verbose:
            print(f'Deleted the following smiles due to nan features/property value:{list(df[smiles][na])}')
        X=self.apply_feature_operation( [feature_dict[k][~na] for k in features])
        y=y_p[~na]  
            
        if multiprop:
            y=np.concatenate([[v] for v in y], axis=0)
                
        return X, y, na 
    
    def get_pairs(self,df,pairs_col:str='pairs'):
        ''' Create list of paired indices for both the original indices and reindexed indices

        Args:
            df: dataframe
            pairs_col: column containing the paired indices
        '''
        assert pairs_col in df.columns
        new_df=df.reset_index()
        old_to_new={old:new for new,old in enumerate(new_df['index'])}
        pair_indices=[]
        pair_ids=[]
        for pl in new_df[pairs_col]:
            for i,j in pl:
                if j in old_to_new:
                    pair_ids.append((i,j))
                    pair_indices.append((old_to_new[i],old_to_new[j]))
        return pair_ids,pair_indices
    
        
               
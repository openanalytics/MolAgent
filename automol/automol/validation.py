"""implementation of the validation set creation.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""

from sklearn.model_selection import train_test_split


import numpy as np

#from .stacking import *
from .stacking import compute_tanimoto_distances
from .stat_plotly_util import plotly_silhouette_scores
from .clustering import MurckoScaffoldClustering, ButinaSplitReassigned, HierarchicalButina, KmeansForSmiles


def stratified_validation(df,properties,stacked_model=None,standard_smiles_column='SMILES',smiles_data=None,test_size=0.25,feature_generators=None,clustering='Bottleneck',
                                                  n_clusters=20,cutoff=0.6,include_chirality=False,
                                                  verbose=0,
                          random_state=42,
                          plot_silhouette=False,
                          cmap='PiYG',
                          fig_size=(600, 600),
                          minority_nb=9,
                         clustering_algorithm=None):
    """
    functions that clusters the data and splits divides the cluster when dividing the data in train and validation data 
    
    Args:
         df: pandas dataframe
         properties:properties
         stacked_model: the FeatureGeneration model that can generates Bottleneck transformer features
         smiles_data: the SMILES
         test_size: the ratio of test versus train
         clustering: string to select clustering options
            clustering='Bottleneck': generate transformer features from smiles and apply Kmeans++
            clustering='Butina': use butina clustering
            clustering='Scaffold': use scaffold clustering 
         n_clusters: number of clusters, value k, in kmeans
         cutoff: the cuttoff used in Butina
         include_chirality: used for scaffold clustering
         verbose: set for more print statements
         random_state: set for reproducability of data splitting when using clustering='Bottleneck'
         plot_silhouette: show the silhouette score
         cmap: colormap for silhouette score
         fig_size: figure size silhouette score
    
    Returns:
        tuple (train, validation, sihl_dict) or (train, validation)
    """
    if feature_generators is None and stacked_model is None:
        raise ValueError(f'feature_generators and stacked_model are None, if no feature_generators are available please provide stacked_model') 
    if feature_generators is None:
        feature_generators=stacked_model.feature_generators
    if smiles_data is None:
        smiles_data=df[standard_smiles_column]

    X=None
    if clustering_algorithm is None: 
        assert clustering in ['Bottleneck','Butina','HierarchicalButina', 'Scaffold'], 'provide one of Bottleneck or Butina or Scaffold'
        if clustering== 'Scaffold':
            clustering_algorithm=MurckoScaffoldClustering(include_chirality=include_chirality)
        elif clustering== 'Butina':
            clustering_algorithm=ButinaSplitReassigned(cutoff = cutoff)
        elif clustering=='HierarchicalButina':
            if cutoff is not None and not isinstance(cutoff, (list)):
                cutoff=[cutoff, np.minimum(cutoff+0.1,np.maximum(cutoff,0.9))]
            clustering_algorithm=HierarchicalButina(cutoff = cutoff)
        elif clustering== 'Bottleneck':
            clustering_algorithm=KmeansForSmiles(n_groups=n_clusters,feature_generators=feature_generators,used_features=['Bottleneck'],random_state=random_state)
            if verbose: 
                print(f'Perform km clustering ({n_clusters} clusters) using the Bottleneck features' )
                
    clustering_algorithm.cluster(smiles_data)
    df['cluster']=clustering_algorithm.get_groups()
    generated_feat=clustering_algorithm.get_generated_features()
    #copy features
    X_list=[]
    for key,item in generated_feat.items():
        X_list.append(item['X'].copy())
    if len(X_list)>0:
        X=np.concatenate( X_list, axis=-1)

    clustering_algorithm.clear_generated_features()   
    
    if plot_silhouette: 
        if X is None: 
            if 'bottleneck' in feature_generators:
                X=feature_generators['bottleneck'].generate(smiles_data)
        sihl_dict=plotly_silhouette_scores(smiles_data,X,df['cluster'],'euclidean',cmap=cmap,fig_size=fig_size,verbose=verbose)
    
    notnull=df[properties].notnull().astype(str).agg('-'.join, axis=1)
    df['stratify'] = notnull+df['cluster'].astype(str) 
    for c in df['stratify'].unique():
        if len(df[df.stratify==c]) < minority_nb:
            df.loc[df.stratify==c, 'stratify']= 'minorities'
    if len(df[df.stratify=='minorities']) == 1:
        distance_to_compounds=compute_tanimoto_distances(df.iloc[np.flatnonzero(df.stratify=='minorities')[0]][standard_smiles_column],df[standard_smiles_column])
        closest_compound=np.argsort(distance_to_compounds)[1]
        df.loc[df.stratify=='minorities', 'stratify']= df.iloc[closest_compound]['stratify']
        
    if verbose: print(df['stratify'].value_counts())
    ### add external datasets if needed. not needed here because they are stored with the model pt file 
    if plot_silhouette: 
        train,validation=train_test_split( df,  test_size=test_size , stratify=df['stratify'].values,random_state =random_state)
        return train, validation, sihl_dict
    return train_test_split( df,  test_size=test_size , stratify=df['stratify'].values,random_state =random_state)



def mixed_validation(df_orig,properties,stacked_model=None,standard_smiles_column='SMILES',prop_cliff_cut=0.6,prop_cliff_butina=0.3,
                     test_size=0.25,feature_generators=None,clustering='Bottleneck',
                     n_clusters=20,cutoff=0.6,include_chirality=False,
                     verbose=0,random_state=42,plot_silhouette=False,cmap='PiYG',
                     fig_size=(600, 600),mix_dict=None, mix_ratios=None,categorical_data=False,minority_nb=9,
                     clustering_algorithm=None,chem_clustering_algorithm=None):
    """
    Splits the data in training and validation set where the validation set is a mix of property cliffs, complete groups and the rest: stratification of the remaining groups.
    
    Args:
         df_orig: dataframe of the data
         properties: list of properties
         stacked_model: the stacking model
         standard_smiles: the list of standardized smiles
         standard_smiles_column:  the column name of the standardized smiles
         prop_cliff_cut: value between 0 and 1 to indicate relative gap for the samples to be considered as property cliff
         prop_cliff_butina: threshold to determine similarity required between samples such that these samples are similar enough to potentially be categorised as property cliff
         test_size: ratio of validation set size wrt to total data size
         clustering: string to select clustering options
            clustering='Bottleneck': generate transformer features from smiles and apply Kmeans++
            clustering='Butina': use butina clustering
            clustering='Scaffold': use scaffold clustering 
         n_clusters: number of clusters, value k, in kmeans
         cutoff: the cuttoff used in Butina
         include_chirality: used for scaffold clustering
         verbose: set for more print statements
         random_state: set for reproducability of data splitting when using clustering='Bottleneck'
         plot_silhouette: show the silhouette score
         cmap: colormap for silhouette score
         fig_size: figure size silhouette score
         mix_dict: dictionary with quantiles for relative count of the different types of validation samples. Keys are the validation types, values the relative count 
         mix_ratios: the relative count for the different types as a list, ignored if mix_dict is given
         categorical_data: boolean to indicate categorical data
    
    Returns:
        tuple with (train,validation,leave group out indices ,property cliff indices dictionary, sihl_dict [optional])
    """
    if feature_generators is None and stacked_model is None:
        raise ValueError(f'feature_generators and stacked_model are None, if no feature_generators are available please provide stacked_model') 
    if feature_generators is None:
        feature_generators=stacked_model.feature_generators
    
    

    if mix_dict is None:
        if mix_ratios is None:
            mix_ratios=[0.15,0.15,0.7]    
        mix_dict={key:value for key,value in zip(['prop_cliffs','leave_group_out','stratified'],mix_ratios)}
    
    test_cnt=test_size*len(df_orig)
    df_orig.reset_index(inplace=True)
    df=df_orig.copy()

    smiles_data=df[standard_smiles_column]

    ###########
    #Activity cliffs
    if mix_dict['prop_cliffs']>0:
        if chem_clustering_algorithm is None:
            chem_clustering_algorithm=ButinaSplitReassigned(cutoff = prop_cliff_butina)
        chem_clustering_algorithm.cluster(df[standard_smiles_column])    
        df['butina_cluster']=chem_clustering_algorithm.get_groups()
        if categorical_data:
            p_cutoff={p: 0.99 for p in properties}
        else:
            p_cutoff={p:prop_cliff_cut*(df[p].max()-df[p].min()) for p in properties}
            #p_cutoff={p:prop_cliff_cut*(np.nanmax(df[p])-np.nanmin(df[p])) for p in properties}
        grouped = df.groupby('butina_cluster')
        activity_cliffs_index={p:[] for p in properties}
        validation_indices=[]
        for name, group in grouped:
            for p in properties:
                for i in range(len(group)):
                    if not np.isfinite(group[p].values[i]):
                        continue
                    adversaries=[]
                    for j in range(i+1, len(group)):
                        if not np.isfinite(group[p].values[j]):
                            continue
                        if abs(group[p].values[i] -group[p].values[j])>p_cutoff[p]:
                            adversaries.append(group['index'].values[j])
                            smi=df.iloc[group['index'].values[i]][standard_smiles_column]
                            adv_smi=df.iloc[group['index'].values[j]][standard_smiles_column]
                            #print(f'smiles: {smi} with prop: {group[p].values[i]} \n and adversary: {adv_smi} with prop: {group[p].values[j]}')
                    if len(adversaries):
                        activity_cliffs_index[p]=activity_cliffs_index[p]+(adversaries+[group['index'].values[i]])
                        break
            temp=[]
            for p in properties:
                temp=temp+activity_cliffs_index[p]
            validation_indices=list(np.unique(validation_indices+temp))
            if len(validation_indices)>mix_dict['prop_cliffs']*test_cnt:
                break
        df.drop(index=validation_indices,inplace=True)
    
    
    ########################
    #clustering for groups
    X=None
    if clustering_algorithm is None: 
        print('Warning: use of strings based clustering choice is deprecated and will be removed in future versions, provide clustering algorithm')
        assert clustering in ['Bottleneck','Butina','HierarchicalButina', 'Scaffold'], 'provide one of Bottleneck or Butina or Scaffold'
        if clustering== 'Scaffold':
            clustering_algorithm=MurckoScaffoldClustering(include_chirality=include_chirality)
        elif clustering== 'Butina':
            clustering_algorithm=ButinaSplitReassigned(cutoff = cutoff)
        elif clustering=='HierarchicalButina':
            if cutoff is not None and not isinstance(cutoff, (list)):
                cutoff=[cutoff, np.minimum(cutoff+0.1,np.maximum(cutoff,0.9))]
            clustering_algorithm=HierarchicalButina(cutoff = cutoff)
        elif clustering== 'Bottleneck':
            clustering_algorithm=KmeansForSmiles(n_groups=n_clusters,feature_generators=feature_generators,used_features=['Bottleneck'],random_state=random_state)
            if verbose: 
                print(f'Perform km clustering ({n_clusters} clusters) using the Bottleneck features' )
                
    clustering_algorithm.cluster(df[standard_smiles_column])
    df['cluster']=clustering_algorithm.get_groups()
    generated_feat=clustering_algorithm.get_generated_features()
    #copy features
    X_list=[]
    for key,item in generated_feat.items():
        X_list.append(item['X'].copy())
    if len(X_list)>0:
        X=np.concatenate( X_list, axis=-1)

    clustering_algorithm.clear_generated_features()  
    
    
    if plot_silhouette: 
        if X is None: 
            if 'bottleneck' in feature_generators:
                X=feature_generators['bottleneck'].generate(smiles_data)
        sihl_dict=plotly_silhouette_scores(df[standard_smiles_column],X,df['cluster'],'euclidean',cmap=cmap,fig_size=fig_size,verbose=verbose)
        
    notnull=df[properties].notnull().astype(str).agg('-'.join, axis=1)
    df['stratify'] = notnull+df['cluster'].astype(str) 
    for c in df['stratify'].unique():
        if len(df[df.stratify==c]) < minority_nb:
            df.loc[df.stratify==c, 'stratify']= 'minorities' 
    if len(df[df.stratify=='minorities']) == 1:
        distance_to_compounds=compute_tanimoto_distances(df.iloc[np.flatnonzero(df.stratify=='minorities')[0]][standard_smiles_column],df[standard_smiles_column])
        closest_compound=np.argsort(distance_to_compounds)[1]
        df.loc[df.stratify=='minorities', 'stratify']= df.iloc[closest_compound]['stratify']
                
    if verbose: print(df['stratify'].value_counts())
    
    #########
    #leave group out selection
    prop_count= df['stratify'].value_counts()
    if mix_dict['leave_group_out']>0:
        min_test_size=mix_dict['leave_group_out']*test_cnt
        test_cluster=[]
        active_test_size=0
        np.random.seed(random_state)
        order=np.random.permutation(range(len(prop_count)))
        lgo_indices=[]
        for i in order:
            if active_test_size+prop_count[i]<min_test_size*1.1:
                active_test_size+=prop_count[i]
                #print(test_size,prop_count[i],df.index[df['cluster'] == i].tolist())
                lgo_indices=lgo_indices+df.index[df['stratify'] == prop_count.index[i]].tolist()
            if active_test_size>min_test_size*0.95:
                break

        lgo_indices=np.sort(lgo_indices)
        df.drop(index=lgo_indices,inplace=True)
    #############
    #stratified validation
    train,test= train_test_split( df, test_size=(test_cnt-len(validation_indices)-len(lgo_indices))/len(df_orig) , stratify=df['stratify'].values,random_state =random_state)
    all_validation=np.sort(list(test['index'].values)+list(validation_indices)+list(lgo_indices))
    validation=df_orig.iloc[all_validation,:]
    indices_map={val:index for index,val in enumerate(validation['index'])}
    train=df_orig.drop(index=all_validation)
    mapped_lgo_indices=[indices_map[val] for val in lgo_indices]
    mapped_activity_cliffs_index={}
    for p in properties:
        mapped_activity_cliffs_index[p]=[indices_map[val] for val in activity_cliffs_index[p]]
    if plot_silhouette: 
        return train,validation,mapped_lgo_indices,mapped_activity_cliffs_index, sihl_dict
    return train,validation,mapped_lgo_indices,mapped_activity_cliffs_index

def leave_grp_out_validation(df,properties,stacked_model=None,standard_smiles_column='SMILES',smiles_data=None,test_size=0.25,
                             feature_generators=None,clustering='Bottleneck',
                                                  n_clusters=20,cutoff=0.6,include_chirality=False,
                                                  verbose=0,random_state=42,plot_silhouette=False,cmap='PiYG',fig_size=(600, 600),clustering_algorithm=None):
    """
    functions that clusters the data and splits the data in train and validation data by addign complete clusters to the validation set
    
    Args:
         df: pandas dataframe
         properties: list of properties
         stacked_model: the FeatureGeneration model that can generates Bottleneck transformer features
         smiles_data: the SMILES
         test_size: the ratio of test versus train
         clustering: string to select clustering options
            clustering='Bottleneck': generate transformer features from smiles and apply Kmeans++
            clustering='Butina': use butina clustering
            clustering='Scaffold': use scaffold clustering 
         n_clusters: number of clusters, value k, in kmeans
         cutoff: the cuttoff used in Butina
         include_chirality: used for scaffold clustering
         verbose: set for more print statements
         random_state: set for reproducability of data splitting when using clustering='Bottleneck'
         plot_silhouette: show the silhouette score
         cmap: colormap for silhouette score
         fig_size: figure size silhouette score
    
    Returns:
        tuple with (train,validation,sihl_dict) or (train,validation)
    """    
    if feature_generators is None and stacked_model is None:
        raise ValueError(f'feature_generators and stacked_model are None, if no feature_generators are available please provide stacked_model') 
    if feature_generators is None:
        feature_generators=stacked_model.feature_generators
    if smiles_data is None:
        smiles_data=df[standard_smiles_column]
    X=None
    if clustering_algorithm is None: 
        assert clustering in ['Bottleneck','Butina','HierarchicalButina', 'Scaffold'], 'provide one of Bottleneck or Butina or Scaffold'
        if clustering== 'Scaffold':
            clustering_algorithm=MurckoScaffoldClustering(include_chirality=include_chirality)
        elif clustering== 'Butina':
            clustering_algorithm=ButinaSplitReassigned(cutoff = cutoff)
        elif clustering=='HierarchicalButina':
            if cutoff is not None and not isinstance(cutoff, (list)):
                cutoff=[cutoff, np.minimum(cutoff+0.1,np.maximum(cutoff,0.9))]
            clustering_algorithm=HierarchicalButina(cutoff = cutoff)
        elif clustering== 'Bottleneck':
            clustering_algorithm=KmeansForSmiles(n_groups=n_clusters,feature_generators=feature_generators,used_features=['Bottleneck'],random_state=random_state)
            if verbose: 
                print(f'Perform km clustering ({n_clusters} clusters) using the Bottleneck features' )
                
    clustering_algorithm.cluster(smiles_data)
    df['cluster']=clustering_algorithm.get_groups()
    generated_feat=clustering_algorithm.get_generated_features()
    #copy features
    X_list=[]
    for key,item in generated_feat.items():
        X_list.append(item['X'].copy())
    if len(X_list)>0:
        X=np.concatenate( X_list, axis=-1)

    clustering_algorithm.clear_generated_features()   
    
    if plot_silhouette: 
        if X is None: 
            if 'bottleneck' in feature_generators:
                X=feature_generators['bottleneck'].generate(smiles_data)
        sihl_dict=plotly_silhouette_scores(smiles_data,X,df['cluster'],'euclidean',cmap=cmap,fig_size=fig_size,verbose=verbose)
        
    prop_count= df['cluster'].value_counts()
    min_test_size=test_size*len(df['cluster'])
    test_cluster=[]
    active_test_size=0
    np.random.seed(random_state)
    order=np.random.permutation(range(len(prop_count)))
    validation_indices=[]
    for i in order:
        if active_test_size+prop_count[i]<min_test_size*1.1:
            active_test_size+=prop_count[i]
            #print(test_size,prop_count[i],df.index[df['cluster'] == i].tolist())
            validation_indices=validation_indices+df.index[df['cluster'] == i].tolist()
        if active_test_size>min_test_size*0.95:
            break
    
    validation_indices=np.sort(validation_indices)
    validation=df.iloc[validation_indices,:]
    train=df
    train=train.drop(index=validation_indices)
    if plot_silhouette: 
        return train,validation,sihl_dict
    return train,validation
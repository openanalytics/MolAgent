"""implementation of the different clustering algorithms and their base class.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""

from typing import List
import numpy as np, pandas as pd
from numpy import linalg as la 

###############################
#Clustering interface
###############################
class ClusteringAlgorithm():
    def __init__(self):
        """!
        Initialization of the base class
        """
        ## groups (array)
        self._groups=[]
        self._sz=-1
        ## list of the names of the features
        self._generated_features={}

    
    def get_groups(self):
        """ Retrieve groups from algorithms

        Returns:
            groups (array): an array filled with cluster/group indices for each smiles

        """
        assert len(self._groups)>0, 'groups not initialised, call cluster'
        return self._groups
    
    def size(self):
        """ Retrieve groups from algorithms

        Returns:
            sz (int): size of the last given list of smiles

        """
        assert len(self._groups)==self.list_sz, 'The size of the given smiles is not equal the generated groups'
        return self._sz
    
    def _check_input_smiles(self,smiles:List[str]):
        """Check provided list of smiles if iterable or pandas series
        
        Args:
            smiles (List[str]): provided list of strings
        
        Attributes:
            sz (int): sz is set to provided length of smiles list
        
        Returns:
            smiles (List[str]): verified/corrected smiles
        """
        try:
            iterator = iter(smiles)
        except TypeError:
            smiles=[smiles]
            
        #pandas.series check
        if hasattr(smiles, 'tolist'):
            smiles = smiles.tolist()
        assert isinstance(smiles, list)
        
        self._sz=len(smiles)
        
        return smiles
    
    def get_generated_features(self):
        """ Retrieve generated features when assigning groups in algorithms
        
        example dictionary contains:
                example_dict={ key:{X:X, cid: generator_name }} with key, the key from the feature_generator, 
                                                                     X,  the 2d numpy matrix and
                                                                     generator_name, the name/version of the feature generator.

        Returns:
            generated_features (dict): return a dictionary with the keys and generated features of the clustering process.
        """
        return self._generated_features
    
    def cluster(self,smiles:List[str]):
        """generate the groups from a given list of smiles
        
        Args:
            smiles (list[str]): list of smiles (list of strings)
        
        Attributes:
            sz (int): length of smiles list
            groups (array): an array filled with cluster/group indices for the smiles list
        """
        pass
    
    def clear_generated_features(self):
        """clears the generated features
        
        Attributes:
            generated_features: set to empty
        """
        self._generated_features={}
    
    

###############################
#MurckoScaffold
###############################
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, MolFromSmiles, AllChem
from rdkit.Chem.rdmolops import FastFindRings
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import  DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.ML.Descriptors import MoleculeDescriptors

class MurckoScaffoldClustering(ClusteringAlgorithm):
    
    def __init__(self,include_chirality:bool=False):
        """Initialisation
        
        Args:
            include_chirality (bool): to include in chiralty in the process
        """
        super(MurckoScaffoldClustering, self).__init__()
        self._include_chirality=include_chirality
        
    def cluster(self,smiles:List[str]):
        """generate the groups from a given list of smiles
        
        Args:
            smiles (list[str]): list of smiles (list of strings)
        
        Attributes:
            sz (int): length of smiles list
            groups (array): an array filled with cluster/group indices for the smiles list
        """
        smiles=self._check_input_smiles(smiles)
        
        molecules=[Chem.MolFromSmiles(s) for s in smiles]
        scaffolds = {}
        for i, mol in enumerate(molecules):
            try:
                FastFindRings(mol)
                mol_scaffold = MurckoScaffoldSmiles( mol=mol, includeChirality=self._include_chirality)
                if mol_scaffold not in scaffolds: scaffolds[mol_scaffold]=[]
                scaffolds[mol_scaffold].append(i)
            except Exception as e:
                if 'Failed' not in scaffolds: scaffolds['Failed']=[]
                scaffolds['Failed'].append(i)
                print(f'Failed to compute the scaffold for molecule {i+1} and it is placed in the failed group. Error: {e}')
        scaffolds2=[]
        self._groups=np.repeat(-1,len(smiles))#groups=[-1 for i in range(len(smiles))]
        for i, c in enumerate(scaffolds):
            scaffolds2.append({'smiles':c,'group':i ,'ids':scaffolds[c]} )
            for e in scaffolds[c]: self._groups[e]=i

            
###############################
#Butina with an extra kmeans based reassignment
###############################

from .stat_util import generate_distance_matrix_lowerdiagonal
from .feature_generators import ECFPGenerator

class ButinaSplitReassigned(ClusteringAlgorithm):
    
    def __init__(self,cutoff:float=0.5, feature_generator: ECFPGenerator= None):
        """Initialization
        
        If no feature_generator is given the default ecfp generator with 1024 bits and radius 2 is used
        
        Args:
            cutoff (float): defines the threshold
            feature_generator (ECFPGenerator): optionally provide feature generator
        """
        super(ButinaSplitReassigned, self).__init__()
        self.set_cutoff(cutoff)
        if feature_generator is None:
            feature_generator= ECFPGenerator(radius=2, nBits =1024)
        self._feat_gen_key=f'fps_{feature_generator.nBits}_{feature_generator.radius}'
        self._feature_generator=feature_generator
        self._center_indices=[]
        
    def set_cutoff(self,cutoff:float=0.5):
        """ set cutoff
        
        Args:
            cutoff (float): threshold to be set
            
        Attributes:
            _cutoff: is set to given value if in [0,1]
        """
        if cutoff is not None and isinstance(cutoff, (list,np.ndarray,tuple)):
            self._cutoff=cutoff[0]
        if cutoff is None:
            cutoff=0.5
        assert cutoff>=0. and cutoff<=1., 'cutoff must be between 0 and 1' 
        self._cutoff=cutoff
        
        
    def cluster(self,smiles:List[str]):
        """generate the groups from a given list of smiles
        
        Args:
            smiles (list[str]): list of smiles (list of strings)
        
        Attributes:
            sz (int): length of smiles list
            groups (array): an array filled with cluster/group indices for the smiles list
        """
        smiles=self._check_input_smiles(smiles)
        
        print(f"Butina clustering with cutoff {self._cutoff}")
        
        fps=self._feature_generator.generate_BitVect(smiles)
        #store features for possible reuse
        self._generated_features={f'{self._feat_gen_key}': {'X':np.array([np.array(fp) for fp in fps]), 'cid': self._feature_generator.get_generator_name()}}
        
        indices=np.array(list(range(len(smiles))),dtype=np.int64)
        self._groups=np.repeat(-1,len(smiles))
        feature_na=np.array([ np.isnan(row).any() for j, row in enumerate(self._generated_features[f'{self._feat_gen_key}']['X'])])
        
        fps=[fp for fp,invalid in zip(fps,feature_na) if not invalid]
        
        
        # scaffold sets
        nfps=len(fps)
        dists = []
        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1 - x for x in sims])
        """
        rdkit documentation: scaffold_sets follows
            a tuple of tuples containing information about the clusters:
                ( (cluster1_elem1, cluster1_elem2, …),
                (cluster2_elem1, cluster2_elem2, …), …

                ) The first element for each cluster is its centroid.
        """
        scaffold_sets = Butina.ClusterData(dists, nfps, self._cutoff, isDistData=True)
        scaffold_sets = sorted(scaffold_sets, key=lambda x: -len(x))
        centers=np.zeros((len(scaffold_sets),self._feature_generator.get_nb_features())) 
        dist_mat=generate_distance_matrix_lowerdiagonal(dists,nfps)
        center_indices=np.zeros(len(scaffold_sets),dtype=np.int64)
        #pick out centroids
        for i, c in enumerate(scaffold_sets):
            centers[i,:]=fps[c[0]]
            center_indices[i]=c[0]
        center_norms=[la.norm(centers[i,:])**2 for i in range(centers.shape[0])]
        groups=np.repeat(-1,nfps)#[-1 for i in range(len(smiles))]
        count=0
        for i, c in enumerate(scaffold_sets):
            for e in c : groups[e]=i
        for i in range(len(smiles)):
            dists_c=centers@fps[i]
            sample_norm= la.norm(fps[i])**2
            tanim_dist=[1-d / (center_norms[j] + sample_norm - d) for j,d in enumerate(dists_c)]
            min_index=np.argmin(tanim_dist)
            if groups[i]!=min_index:
                count+=1
            groups[i]=min_index

        print(f'Reassigned {count} samples to different cluster')
        unique_classes =np.unique(groups)
        if len(unique_classes) != len(scaffold_sets):
            for i in range(len(smiles)):
                val=self._groups[i]
                assigned_cluster=np.where(unique_classes == val)[0]
                groups[i]=assigned_cluster
        
        #keep nans as -1
        self._groups[~feature_na]=groups
        self._center_indices=indices[~feature_na][center_indices]
    
    def get_center_indices(self):
        """ Retrieve center_indices

        Returns:
            center_indices (array): indices of the centers

        """
        assert len(self._center_indices)>0, 'Center indices not initialised, call cluster'
        return self._center_indices
    
    
class HierarchicalButina(ClusteringAlgorithm):
    
    def __init__(self,cutoff:List[float]=[0.6,0.7], feature_generator: ECFPGenerator= None):
        
        """Initialization
        
        If no feature_generator is given the default ecfp generator with 1024 bits and radius 2 is used
        
        Args:
            cutoff (List[float]): defines the list of thresholds
            feature_generator (ECFPGenerator): optionally provide feature generator
        """
        super(HierarchicalButina, self).__init__()
        if cutoff is not None and not isinstance(cutoff, (list,np.ndarray,tuple)):
            cutoff=[cutoff]
        self._cutoff=cutoff
        self._butina_clustering=ButinaSplitReassigned(cutoff=cutoff[0],feature_generator=feature_generator)
    
    
    def cluster(self,smiles):
        """"generate the groups from a given list of smiles
        
        Args:
            smiles (list[str]): list of smiles (list of strings)
        
        Attributes:
            sz (int): length of smiles list
            groups (array): an array filled with cluster/group indices for the smiles list
        """
        
        smiles=self._check_input_smiles(smiles)

        clustered_groups = np.arange(len(smiles))
        clustered_smiles  = np.array(smiles)
        for index,coff in enumerate(self._cutoff):
            self._butina_clustering.set_cutoff(coff)
            self._butina_clustering.cluster(clustered_smiles)
            groups = np.array(self._butina_clustering.get_groups())
            center_indices = self._butina_clustering.get_center_indices()
            #store generated features at the first level
            if index==0:
                self._generated_features=self._butina_clustering.get_generated_features()
            clustered_smiles = clustered_smiles[center_indices]
            clustered_groups = groups[clustered_groups]
            
        self._groups=clustered_groups
        
        return clustered_groups
    
###############################
#Sklearn kmeans++
###############################    
from sklearn.cluster import KMeans as skKMeans
from .feature_generators import retrieve_default_offline_generators, ECFPGenerator

class KmeansForSmiles(ClusteringAlgorithm):
    
    def __init__(self, n_groups:int=30,*, feature_generators: dict= {},used_features:List[str]=None, random_state:int=42):
        """initialisation of KmeansForSmiles with provided dictionary of feature generators and list of used features.
        
        If feature generation dictionary is not provided, default public generators are used
        
        If used features is not provided or provided features are not available in the generator dictionary,
                Bottleneck is used if present in generation dictionary,
                otherwise al generators in the dictionary are used. 
        
        Args:
            n_groups (int): number of requested groups
            feature_generators(dict): dictionary containing different feature generators
            used_features(list[str]): list of keys indicating the used features
        """
        super(KmeansForSmiles, self).__init__()
        if not feature_generators or feature_generators is None:
            self._feature_generators= retrieve_default_offline_generators(model='ChEMBL', radius=2, nbits=2048, chylearn=0)
        else:
            assert isinstance(feature_generators,dict), 'provided feature generators must be dictionary'
            self._feature_generators=feature_generators
            
        if used_features is None or len(used_features)<1:
            used_features=['Bottleneck']
        self._used_features=[]
        for feat in used_features:
            if feat in self._feature_generators:
                self._used_features.append(feat)
            elif feat.startswith('fps'):
                splits=feat.split('_')
                self._feature_generators[feat]=ECFPGenerator(radius=splits[2], nBits =splits[1])
                self._used_features.append(feat)
        if len(self._used_features)<1:
            if 'Bottleneck' in self._feature_generators:
                self._used_features.append('Bottleneck')
            else:
                for key,item in self._feature_generators.items():
                    self._used_features.append(key)
        
        assert n_groups>0, 'provided num of groups must be larger than zero'
        self._n_groups=n_groups
        self._random_state=random_state
        
            
    
    
    def cluster(self,smiles):
        """"generate the groups from a given list of smiles
        
        Args:
            smiles (list[str]): list of smiles (list of strings)
        
        Attributes:
            sz (int): length of smiles list
            groups (array): an array filled with cluster/group indices for the smiles list
            generated_features(dict): dictionary with all the generated feature matrices
        """
        
        smiles=self._check_input_smiles(smiles)
        self._groups=np.repeat(-1,len(smiles))
        
        kmeans_sk = skKMeans(init="k-means++",n_clusters=self._n_groups,random_state=self._random_state)#,n_jobs=n_jobs)
        X_list=[]
        for key in self._used_features:
            self._generated_features[key]= {'X': self._feature_generators[key].generate(smiles), 'cid': self._feature_generators[key].get_generator_name() }
            X_list.append(self._generated_features[key]['X'])
        X_train=np.concatenate( X_list, axis=-1)
        indices=np.array(list(range(len(smiles))))
        self._groups=np.repeat(-1,len(smiles))
        feature_na=np.array([ np.isnan(row).any() for j, row in enumerate(X_train)])
        
        groups=kmeans_sk.fit(X_train[~feature_na]).labels_
        self._groups[~feature_na]=groups
        

###################################
# Generic clustering using provided sklearn estimator
###################################
class SklearnClusteringForSmiles(ClusteringAlgorithm):
    
    def __init__(self,*, feature_generators: dict= {},used_features:List[str]=None, random_state:int=42,sklearn_estimator=None):
        """initialisation of KmeansForSmiles with provided dictionary of feature generators and list of used features.
        
        If feature generation dictionary is not provided, default public generators are used
        
        If used features is not provided or provided features are not available in the generator dictionary,
                Bottleneck is used if present in generation dictionary,
                otherwise al generators in the dictionary are used. 
        
        Args:
            feature_generators(dict): dictionary containing different feature generators
            used_features(list[str]): list of keys indicating the used features
            sklearn_estimator: provided clustering algorithm
        """
        super(KmeansForSmiles, self).__init__()
        if not feature_generators or feature_generators is None:
            self._feature_generators= retrieve_default_offline_generators(model='ChEMBL', radius=2, nbits=2048, chylearn=0)
        else:
            assert isinstance(feature_generators,dict), 'provided feature generators must be dictionary'
            self._feature_generators=feature_generators
            
        if used_features is None or len(used_features)<1:
            used_features=['Bottleneck']
        self._used_features=[]
        for feat in used_features:
            if feat in self._feature_generators:
                self._used_features.append(feat)
            elif feat.startswith('fps'):
                splits=feat.split('_')
                self._feature_generators[feat]=ECFPGenerator(radius=splits[2], nBits =splits[1])
                self._used_features.append(feat)
        if len(self._used_features)<1:
            if 'Bottleneck' in self._feature_generators:
                self._used_features.append('Bottleneck')
            else:
                for key,item in self._feature_generators.items():
                    self._used_features.append(key)
        
        self._random_state=random_state
        self._estimator=sklearn_estimator
            
    
    
    def cluster(self,smiles):
        """"generate the groups from a given list of smiles
        
        Args:
            smiles (list[str]): list of smiles (list of strings)
        
        Attributes:
            sz (int): length of smiles list
            groups (array): an array filled with cluster/group indices for the smiles list
            generated_features(dict): dictionary with all the generated feature matrices
        """
        
        smiles=self._check_input_smiles(smiles)
        self._groups=np.repeat(-1,len(smiles))
        
        X_list=[]
        for key in self._used_features:
            self._generated_features[key]= {'X': self._feature_generators[key].generate(smiles), 'cid': self._feature_generators[key].get_generator_name() }
            X_list.append(self._generated_features[key]['X'])
        X_train=np.concatenate( X_list, axis=-1)
        indices=np.array(list(range(len(smiles))))
        self._groups=np.repeat(-1,len(smiles))
        feature_na=np.array([ np.isnan(row).any() for j, row in enumerate(X_train)])
        
        groups=self._estimator.fit_predict(X_train[~feature_na])
        self._groups[~feature_na]=groups

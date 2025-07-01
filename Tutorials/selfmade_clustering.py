'''
Your own clustering class should inherit from

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
        """ Retrieve groups from algorithms
        
        eample dictionary contains:
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

In essence you only have to implement the init and cluster functionality. The cluster function takes as input a list/series of smiles and sets the 
attribute _groups. This is essential. The groups are the assigned cluster indices. 

Your code can ignore nan features and assign them to a failed class, the nan features are filtered out when actually training a model. 

When generating features you can save them in the feature_generators attribute following the convention 
    example_dict={ key:{X:X, cid: generator_name }} with key, the key from the feature_generator, 
                                                         X,  the 2d numpy matrix and
                                                         generator_name, the name/version of the feature generator.
    These are reused when the algorithm is used to define the groups in nested-cross-validation. The code retrieves the matrices and checks that the provided generator name
    matches the generator name of the feature generators of the model (using the provided key) 
'''

from automol.clustering import ClusteringAlgorithm
from automol.feature_generators import retrieve_default_offline_generators, ECFPGenerator
from typing import List
import numpy as np

class SklearnClusteringForSmiles(ClusteringAlgorithm):
    
    def __init__(self,*, feature_generators: dict= {},used_features:List[str]=None, random_state:int=42,sklearn_estimator=None):
        """initialisation of KmeansForSmiles with provided dictionary of feature generators and list of used features.
        
        If feature generation dictionary is not provided, default public generators are used
        
        If used features is not provided or provided features are not available in the generator dictionary,
                DL is used if present in generation dictionary,
                otherwise al generators in the dictionary are used. 
        
        Args:
            feature_generators(dict): dictionary containing different feature generators
            used_features(list[str]): list of keys indicating the used features
        """
        super(SklearnClusteringForSmiles, self).__init__()
        if not feature_generators or feature_generators is None:
            self._feature_generators= retrieve_default_offline_generators(model='ChEMBL', radius=2, nbits=2048, chylearn=0)
        else:
            assert isinstance(feature_generators,dict), 'provided feature generators must be dictionary'
            self._feature_generators=feature_generators
            
        if used_features is None or len(used_features)<1:
            used_features=['DL']
        self._used_features=[]
        for feat in used_features:
            if feat in self._feature_generators:
                self._used_features.append(feat)
            elif feat.startswith('fps'):
                splits=feat.split('_')
                self._feature_generators[feat]=ECFPGenerator(radius=splits[2], nBits =splits[1])
                self._used_features.append(feat)
        if len(self._used_features)<1:
            if 'DL' in self._feature_generators:
                self._used_features.append('DL')
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

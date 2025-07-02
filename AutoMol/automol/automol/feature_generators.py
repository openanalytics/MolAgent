"""implementation of the different feature generators and their base class.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""

from .model import *

import pandas as pd

import numpy as np

import torch
from rdkit import __version__ as rdkit_version
from rdkit import Chem
from rdkit.Chem import Descriptors, MolFromSmiles, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

    
from importlib_resources import files


def retrieve_default_offline_generators(model='CHEMBL', radius=2, nbits=2048):
    """
    Function that returns a dictionary of default internal feature generators.
    
    Args:
        model: string that to define which kind of Deeplearning generated features. Reflects the smiles used for training the encoder.
        radius: radius of ecfp generation
        nbits: size of the ecfp features
    """

    return {'Bottleneck':BottleneckTransformer(model='CHEMBL'),
            'rdkit':RDKITGenerator(),
            f'fps_{nbits}_{radius}':ECFPGenerator(radius=radius, nBits =nbits)
           }

###############################
class FeatureGenerator():
    def __init__(self):
        """
        Initialization of the base class
        """
        ## number of features
        self.nb_features=-1
        ## list of the names of the features
        self.names=[]
        ## the name of the generator
        self.generator_name=''
    
    def get_nb_features(self):
        """
        getter for the number of features.
        
        includes an assert that number of features is positive.
        
        Returns
            nb_features (int): number of features
        """
        assert self.nb_features>0, 'method not correctly created, negative number of features'
        return self.nb_features
    
    def check_consistency(self):
        """
        checks if the number of features is positive and the length of the feature names equal the number of features
        """
        assert len(self.names)==self.nb_features, 'Provided number of names is not equal to provided number of features'
        assert self.nb_features>0, 'negative number of features'
    
    def generate(self,smiles):
        """
        generate the feature matrix from a given list of smiles
        
        Args:
            smiles: list of smiles (list of strings)
        
        Returns:
            X: feature matrix as numpy array 
        """
        pass
    
    def generate_w_pairs(self,smiles,original_indices,new_indices):
        """
        generate the feature matrix from a given list of smiles
        
        Args:
            smiles: list of smiles (list of strings)
            original_indices: indices for pairs of ligands without reindexing after datasplitting
            new_indices: list indices for pairs of ligands with reindexing after datasplitting
        Returns:
            X: feature matrix as numpy array 
        """
        X=self.generate(smiles)
        X_p=np.zeros((len(new_indices),2*X.shape[1]))           
        for idx,(i,j) in enumerate(new_indices):
            X_p[idx,:]=np.hstack((X[i,:],X[j,:]))
        return X_p
    
    def get_names(self):
        """
        getter for the names of the features
        
        Returns:
            names (List[str]): list of names
        """
        return self.names
    
    def get_generator_name(self):
        """
        getter for the generator name
        
        Returns:
            generator_name (str): the name of the generator
        """
        return self.generator_name

###############################    
class RDKITGenerator(FeatureGenerator):
    """
    feature generator returning the rdkit descriptors
    """

    def __init__(self):
        """
        Initialization 
        """
        ## list of rdkit names from rdkit.Chem.Descriptors.descList
        self.rdkitnames=[ n for n,f in Descriptors.descList]
        ## descriptor calculator MoleculeDescriptors.MolecularDescriptorCalculator(self.rdkitnames)
        self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(self.rdkitnames)
        ## list of names of the features
        self.names= self.calculator.GetDescriptorNames()
        ## number of features
        self.nb_features=len(self.rdkitnames)
        ## generator name
        self.generator_name=f'automol_rdkit_{rdkit_version}'
        
    def get_descriptor(self,s):
        """
        retrieve rdkit descriptors for given smiles s
        
        return tuple of nans if the rdkit fails to calculate descriptors
        
        Args:
            s (str): smiles string
        
        Returns:
            rdkit descriptors or nans
        """
        if s=="" or s is None:
            return self.nb_features*(np.nan,)
        try:
            m=MolFromSmiles(s)
            if m :
                return self.calculator.CalcDescriptors(m)
            else:
                return  self.nb_features*(np.nan,)
        except:
            return self.nb_features*(np.nan,)
        
    def generate(self,smiles):
        """
        Generate all given descriptors for given list of smiles and return as numpy array
        
        Args:
            smiles: list of smiles (list of strings)
        
        Returns:
            des: feature matrix as numpy array 
        """
        des = np.array([self.get_descriptor(x) for x in smiles])
        #test nan 
        #des[-1]=np.array(self.nb_features*(np.nan,))
        return des
    
###############################
class ECFPGenerator(FeatureGenerator):
    """
    The chemical fingerprints generator using rdkit
    """

    def __init__(self,radius=2, nBits =2048,useChirality= False,useFeatures= False):
        """
        Initialization of the ecfp generator
        
        see rdkit.AllChem.GetMorganFingerprintAsBitVect for details on the morgan fingerprint generation
        
        Args:
            radius: radius for morgan fingerprints [=2]
            nBits: number of bits used [=2048]
            useChirality: boolean to set to use chirality when computing fps[=False]
            useFeatures: boolean to set [=False]
        """
        ## radius
        self.radius=int(radius)
        ## nbits
        self.nBits=int(nBits)
        ## boolean to indicated use of chirality when computing fps
        self.useChirality=useChirality
        ## boolean to indicated use of features when computing fps
        self.useFeatures=useFeatures
        ## number of features
        self.nb_features=int(nBits)
        ## list of feature names
        self.names=[f'fps_{i}_of_{nBits}_radius_{radius}' for i in range(int(nBits))]
        ## generator name
        self.generator_name=f'automol_ecfp_{nBits}_radius_{radius}_rdkit_{rdkit_version}'
    
    def generate(self,smiles):
        """
        Generate ecfp for given list of smiles and return as numpy array
        
        Args:
            smiles: list of smiles (list of strings)
        
        Returns:
            X: feature matrix as numpy array
        """
        #mols =[Chem.MolFromSmiles(s) for s in smiles]
        
        return np.array([np.array(fps) for fps in self.generate_BitVect(smiles)],dtype=float)
        #test nan 
        #outputs=np.array([np.array(fps) for fps in self.generate_BitVect(smiles)],dtype=float)
        #outputs[-1]=np.array(self.nb_features*(0,))
        #return outputs
    
    def generate_BitVect(self,smiles):
        """generate the features as BitVects 
        
        Args:
            smiles (list[str]): list of smiles (list of strings)
        
        Returns:
            The list of bitVect belonging to the given smiles
        """
        def getFP(s):
            try:
                return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), radius=self.radius,
                                                     nBits=self.nBits,
                                                     useChirality=self.useChirality,useFeatures=self.useFeatures)
            except:
                return self.nBits*[np.nan]
        return [getFP(s) for s in smiles]
                    

###############################
class BottleneckTransformer(nn.Module,FeatureGenerator):
    '''
    Feature generator returning the features of the bottleneck transformer
    '''
    def __init__(self,model='CHEMBL' ,use_gpu=False,batch_size=100,seq_len=220,model_file=None):
        '''
        initialization that loads the Bottleneck model
        
        Args:
            model:string indicating use of encoder [ChEMBL]
            use_gpu: boolean to use gpu
            batch_size: integer batch size of processed smiles
            seq_len: max sequence length of the smiles
            model_file: model file of the encoder [=None], if None internal model files are used based on model param. 
        '''
        super(BottleneckTransformer, self).__init__()
        assert   model in ['CHEMBL'], 'provide valid encoder'
        ## model file of the encoder
        self.base_model_file=None
        if use_gpu and  torch.cuda.is_available():
            device = torch.device(f'cuda:0')
        else:
            device =torch.device('cpu')
        if model_file is None: 
            self.base_model_file =  str(files('automol.trained_models').joinpath('bottleneck_CHEMBL27_ENUM_SMILES_ENCODER.pt')) 
        else:
            self.base_model_file=model_file
        print(f'loading the base model from file: {self.base_model_file}')
        checkpoint = torch.load(self.base_model_file , map_location='cpu',weights_only=False)
        ## smiles encoder
        self.Smiles_Encoder= checkpoint['Smiles_Encoder']
        self.Smiles_Encoder.load_state_dict(checkpoint['Smiles_Encoder_model_state_dict'])
        ## vocabulary
        self.vocab= checkpoint['vocab']
        self.pad_index= self.vocab.pad_index
        self.Smiles_Encoder.eval()
        self.Smiles_Encoder=self.Smiles_Encoder.to(device)
        ## batch size
        self.batch_size=batch_size
        ## sequence length
        self.seq_len=seq_len
        ## hard coded number of features
        self.nb_features=250
        ## list of feature names
        self.names=[f'Bottleneck_{i}_of_{self.nb_features}_model_{model}' for i in range(self.nb_features)]
        ## generator name
        self.generator_name=f'automol_bn_{self.base_model_file}'
        print('using device:',device)        
        
    def makebatch(self,sms, seq_len=220):
        """
        function that create a batch that is fed to the encoder
        
        Args:
            sms: list of smiles
            seq_len: max sequence length of the smiles
            
        Returns:
            torch tensor with the tokenized values
        """
        intgs=[]
        for s in sms:
            i= self.vocab.smile2int( s, max_smile_len=seq_len, with_eos=True, with_sos=True, return_len=False)
            intgs.append(torch.tensor(i))
        out=torch.stack(intgs, dim=0)
        return out
                    
    def generate(self, smiles):
        '''
        generates the features
        
        Args:
            smiles: list of smiles
        
        Returns:
            numpy matrix with the generated feature matrix X
        '''
        dataframe=False
        if isinstance(smiles, pd.DataFrame) or isinstance(smiles, pd.Series):
            dataframe=True
        last_en_layer=self.Smiles_Encoder.encoder.num_layers-1
        device=next(self.Smiles_Encoder.parameters()).device
        self.Smiles_Encoder.eval()
        outputs=[]
        st=0
        end=0
        while end < (len(smiles)):
            end= min(st+self.batch_size, len(smiles))
            if dataframe:
                src=self.makebatch(smiles.iloc[st:end] ,seq_len=self.seq_len)
            else:
                src=self.makebatch(smiles[st:end] ,seq_len=self.seq_len)
            st+=self.batch_size
            src = src.to(device)
            src= torch.t(src)
            with torch.no_grad():
                out=self.Smiles_Encoder(src)[f'{last_en_layer}']['output'][0, :]
                #print('out shape',out.shape)
            outputs.append(out.detach().cpu().numpy() )
        outputs = np.concatenate(outputs, axis=0)
        #test nan 
        #outputs[-1]=np.array(self.nb_features*(np.nan,))
        return outputs
    
    def __call__(self, smiles,batch_size=100,seq_len=220):
        """
        operator() to generate features, call function generate
        
        Args:
            smiles: list of smiles
            batch_size: batch size
            seq_len: max sequence length
        
        Returns:
            numpy matrix containing the generated features
        """
        self.batch_size=batch_size
        self.seq_len=seq_len
        return self.generate( smiles)
    


class MolfeatGenerator(FeatureGenerator):
    def __init__(self):
        super().__init__()
    
    def generate(self, smiles):
        self.check_consistency()
        st=0
        end=0
        X_list=[]
        while end < (len(smiles)):
            end= min(st+self.batch_size, len(smiles))
            smiles_l=smiles[st:end]
            features = np.full([len(smiles_l), self.nb_features], np.nan)
            indices = []
            structures = []
            for i, s in enumerate(smiles_l):
                if s is None or s=='':
                    continue
                try:
                    m=Chem.MolFromSmiles(s)
                except Exception:
                    continue
                if m is not None:
                    indices.append(i)
                    structures.append(Chem.MolToSmiles(m))

            if structures:
                features[indices] = np.stack(self.model(structures))
            
            st+=self.batch_size
            X_list.append(features)
        return np.concatenate(X_list, axis=0)

class MolfeatPretrainedHFTransformer(MolfeatGenerator):
    def __init__(self, kind='MolT5', notation='smiles', dtype=float,max_length=220,batch_size=250):
        from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
        
        super().__init__()
        self.model = PretrainedHFTransformer(kind=kind, notation=notation, dtype=dtype,max_length=max_length)
        
        X_try=np.stack(self.model(['Oc1ccc(cc1OC)C=O']))
        self.nb_features=X_try.shape[1]
        self.generator_name = f'automol_PretrainedHFTransformer_{kind}'
        self.names.extend(f'feature_{x}' for x in range(self.nb_features))
        self.batch_size=batch_size


class MolfeatFPVecTransformer(MolfeatGenerator):
    def __init__(self, kind='desc2D', dtype=float,batch_size=250):
        from molfeat.trans.fp import FPVecTransformer
        
        super().__init__()
        self.model = FPVecTransformer(kind=kind, dtype=dtype)
        
        X_try=np.stack(self.model(['Oc1ccc(cc1OC)C=O']))
        self.nb_features=X_try.shape[1]
        self.generator_name = f'automol_FPVecTransformer_{kind}'
        self.names.extend(f'feature_{x}' for x in range(self.nb_features))
        self.batch_size=batch_size
        
class Molfeat3DFPVecTransformer(MolfeatGenerator):
    def __init__(self, kind='desc2D', dtype=float,batch_size=250,seed=42):
        from molfeat.trans.fp import FPVecTransformer
        
        super().__init__()
        self.model = FPVecTransformer(kind=kind, dtype=dtype)
        self._seed=seed
        
        m = Chem.MolFromSmiles('Oc1ccc(cc1OC)C=O')
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m, randomSeed=self._seed)
        X_try=np.stack(self.model([m]))
        self.nb_features=X_try.shape[1]
        self.generator_name = f'automol_FPVecTransformer_{kind}'
        self.names.extend(f'feature_{x}' for x in range(self.nb_features))
        self.batch_size=batch_size
        
    def generate(self, smiles):
        self.check_consistency()
        st=0
        end=0
        X_list=[]
        while end < (len(smiles)):
            end= min(st+self.batch_size, len(smiles))
            smiles_l=smiles[st:end]
            features = np.full([len(smiles_l), self.nb_features], np.nan)
            indices = []
            structures = []
            for i, s in enumerate(smiles_l):
                if s is None or s=='':
                    continue
                try:
                    m = Chem.MolFromSmiles(s)  # talidomide
                    m = Chem.AddHs(m)
                    AllChem.EmbedMolecule(m, randomSeed=self._seed)
                except Exception:
                    continue
                if m is not None:
                    indices.append(i)
                    structures.append(m)

            if structures:
                features[indices] = np.stack(self.model(structures))

            st+=self.batch_size
            X_list.append(features)
        return np.concatenate(X_list, axis=0)


class MolfeatMoleculeTransformer(MolfeatGenerator):
    def __init__(self, featurizer='mordred', dtype=float,batch_size=250):
        from molfeat.trans import MoleculeTransformer
        
        super().__init__()
        self.model = MoleculeTransformer(featurizer=featurizer, dtype=dtype)
        
        X_try=np.stack(self.model(['Oc1ccc(cc1OC)C=O']))
        self.nb_features=X_try.shape[1]
        if isinstance(featurizer,str):
            self.generator_name = f'automol_MoleculeTransformer_{featurizer}'
        else:
            self.generator_name = f'automol_MoleculeTransformer'
        self.names.extend(f'feature_{x}' for x in range(self.nb_features))
        self.batch_size=batch_size
        
class Molfeat3DMoleculeTransformer(MolfeatGenerator):
    def __init__(self, featurizer='mordred', dtype=float,batch_size=250,seed=42):
        from molfeat.trans import MoleculeTransformer
        
        super().__init__()
        self._seed=seed
        self.model = MoleculeTransformer(featurizer=featurizer, dtype=dtype)
        
        m = Chem.MolFromSmiles('Oc1ccc(cc1OC)C=O')
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m, randomSeed=self._seed)
        X_try=np.stack(self.model([m]))
        self.nb_features=X_try.shape[1]
        if isinstance(featurizer,str):
            self.generator_name = f'automol_3dMoleculeTransformer_{featurizer}'
        else:
            self.generator_name = f'automol_3dMoleculeTransformer'
        self.names.extend(f'feature_{x}' for x in range(self.nb_features))
        self.batch_size=batch_size        
        
    def generate(self, smiles):
        self.check_consistency()
        st=0
        end=0
        X_list=[]
        while end < (len(smiles)):
            end= min(st+self.batch_size, len(smiles))
            smiles_l=smiles[st:end]
            features = np.full([len(smiles_l), self.nb_features], np.nan)
            indices = []
            structures = []
            for i, s in enumerate(smiles_l):
                if s is None or s=='':
                    continue
                try:
                    m = Chem.MolFromSmiles(s)  # talidomide
                    m = Chem.AddHs(m)
                    AllChem.EmbedMolecule(m, randomSeed=self._seed)
                except Exception:
                    continue
                if m is not None:
                    indices.append(i)
                    structures.append(m)

            if structures:
                features[indices] = np.stack(self.model(structures))
            
            st+=self.batch_size
            X_list.append(features)
        return np.concatenate(X_list, axis=0)
    
        

class MolfeatPretrainedDGLTransformer(MolfeatGenerator):
    def __init__(self, kind='gin_supervised_edgepred', dtype=float,batch_size=250):
        from molfeat.trans.pretrained import PretrainedDGLTransformer
        
        super().__init__()
        self.model =  PretrainedDGLTransformer(kind=kind, dtype=dtype)
        
        X_try=np.stack(self.model(['Oc1ccc(cc1OC)C=O']))
        self.nb_features=X_try.shape[1]
        self.generator_name = f'automol_MPretrainedDGLTransformer_{kind}'
        self.names.extend(f'feature_{x}' for x in range(self.nb_features))
        self.batch_size=batch_size

    
class MolfeatGraphormerTransformer(MolfeatGenerator):
    def __init__(self, kind='pcqm4mv2_graphormer_base', dtype=float,batch_size=250):
        from molfeat.trans.pretrained import GraphormerTransformer
        
        super().__init__()
        self.model =  GraphormerTransformer(kind=kind, dtype=dtype)
        
        X_try=np.stack(self.model(['Oc1ccc(cc1OC)C=O']))
        self.nb_features=X_try.shape[1]
        self.generator_name = f'automol_GraphormerTransformer_{kind}'
        self.names.extend(f'feature_{x}' for x in range(self.nb_features))
        self.batch_size=batch_size

    

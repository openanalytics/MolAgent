import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os, sys ,inspect
#from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import  DataLoader
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, ZINC, GNNBenchmarkDataset
from torch_geometric.utils import degree
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
#from utils import process_sph
from torch.utils.data import  DataLoader, Dataset
import pyximport
add_path=os.path.realpath(__file__)
#sys.path.append(add_path)
pyximport.install(setup_args={'include_dirs': np.get_include()})
#from add_path import algos
import algos


#from commute import adj_pinv
#from sklearn.metrics.pairwise import cosine_similarity

#####################

from data_util import *
from torch_geometric.data import InMemoryDataset


import os
import os.path as osp
############################################

class Compose_and_cat(BaseTransform):
    r"""Composes several transforms and cat them together.

    Args:
        transforms (List[Callable]): List of transforms to compose.
    """
    def __init__(self, transforms,
                 attr_name='pe'
                ):
        self.transforms = transforms
        self.attr_name=attr_name

    def forward(    self,    data ) :
        for transform in self.transforms:
            if isinstance(data, (list, tuple)):
                data = [self.tranform_one_data(d) for d in data]
            else:
                data = self.tranform_one_data(data)
        return data
    def tranform_one_data(self, data):
        for transform in self.transforms:
            data = transform(data)
        catted= [data[transform.attr_name] for  transform in self.transforms]  
        data[self.attr_name] = torch.cat(catted, dim=-1)
        return data

    def __repr__(self) -> str:
        args = [f'  {transform}' for transform in self.transforms]
        return 'cat_pe_{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))
################################


##########################
class online_gps_Dataset(torch.utils.data.Dataset):
    #def __init__(self, smiles, smiles2graph,pre_transform=None ):
    def __init__(self, df=None, df_file=None, smiles='smiles',properties= ['A','B'] , smiles2graph=smiles2graph,
                 transform=None, pre_transform=None):
        super().__init__()
    
        self.df_file = df_file
        self.df = df
        if self.df_file: 
            self.df = pd.read_csv(self.df_file)
        df.reset_index(drop=True, inplace=True)
        assert isinstance(self.df,pd.core.frame.DataFrame), 'provide pandas dataframe'
        assert smiles in self.df.columns , f'{smiles} is not in the columns of {self.df}'
        self.smiles= smiles
        self.properties=properties
        for pro in self.properties:
            assert pro in self.df.columns , f'{pro} is not in the columns of df'
        self.smiles2graph = smiles2graph 
        self.pre_transform=pre_transform
    def __getitem__(self, index):
        s=self.df.iloc[index][self.smiles]
        #print(index, s)
        data = Data()
        graph = self.smiles2graph(s)
        assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert (len(graph['node_feat']) == graph['num_nodes'])
        data.__num_nodes__ = int(graph['num_nodes'])
        data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        N = data.x.shape[0]
        adj = torch.zeros([N, N])
        adj[data.edge_index[0, :], data.edge_index[1, :]] = True
        sp, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
        data['sph']=sp
        if self.pre_transform: data=self.pre_transform(data)
        for pro in self.properties: data[pro]=self.df.iloc[index][pro]
        return data

    def __len__(self):
        return len(self.df)
#######################
#this data set is used by the model for infrence 
##########################
class infer_Dataset(torch.utils.data.Dataset):
    def __init__(self, smiles, smiles2graph,pre_transform=None ):
        super().__init__()
        self.smiles = smiles
        self.smiles2graph=smiles2graph
        self.pre_transform=pre_transform
        
    def __getitem__(self, index):
        s=self.smiles[index]
        data = Data()
        graph = self.smiles2graph(s)
        assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert (len(graph['node_feat']) == graph['num_nodes'])
        data.__num_nodes__ = int(graph['num_nodes'])
        data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        N = data.x.shape[0]
        adj = torch.zeros([N, N])
        adj[data.edge_index[0, :], data.edge_index[1, :]] = True
        sp, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
        data['sph']=sp
        if self.pre_transform: data=self.pre_transform(data)
        return data

    def __len__(self):
        return len(self.smiles)
def prepare_batch_w_sph(data_list):
    max_num_nodes = max([data.sph.shape[0] for data in data_list])
    for data in data_list:
        num_nodes = data.num_nodes
        pad_size = max_num_nodes - num_nodes
        data.sph = torch.nn.functional.pad(data.sph, (0, pad_size, 0, pad_size), value=510)
    batched_data = Batch.from_data_list(data_list)
    return batched_data


###########################################################################
def fn(data_list):
    max_num_nodes = max([data.sph.shape[0] for data in data_list])
    for data in data_list:
        num_nodes = data.num_nodes
        pad_size = max_num_nodes - num_nodes
        data.sph = torch.nn.functional.pad(data.sph, (0, pad_size, 0, pad_size), value=510)
    batched_data = Batch.from_data_list(data_list)
    return batched_data
"""    
def Smiles2GraphData(smiles):
    data_list = []
    for s in smiles:    
        data = Data()
        graph = smiles2graph(s)
        assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert (len(graph['node_feat']) == graph['num_nodes'])
        data.__num_nodes__ = int(graph['num_nodes'])
        data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        N = data.x.shape[0]
        adj = torch.zeros([N, N])
        adj[data.edge_index[0, :], data.edge_index[1, :]] = True
        sp, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
        data['sph']=sp
        data_list.append(data)
    max_num_nodes = max([data.sph.shape[0] for data in data_list])
    for data in data_list:
        num_nodes = data.num_nodes
        pad_size = max_num_nodes - num_nodes
        data.sph = torch.nn.functional.pad(data.sph, (0, pad_size, 0, pad_size), value=510)
    batched_data = Batch.from_data_list(data_list)
    return batched_data   
"""
#######################
class NewDataset_w_sph(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.dataset = original_dataset
        self.sph = []
        self.get_sph_all()
        assert len(self.sph)== len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        data['sph'] = self.sph[index]
        return data

    def __len__(self):
        return len(self.dataset)
    def get_sph_all(self):
        self.sph = []
        file = osp.join('/'.join(self.dataset.processed_paths[0].split('/')[:-1]), 'sph.pkl')
        if not os.path.exists(file):
            print('pre-process sph start!')
            progress_bar = tqdm(desc='pre-processing Data', total=len(self.dataset), ncols=40)
            for i in range(len(self.dataset)):
                self.process(i)
                progress_bar.update(1)
            progress_bar.close()
            pickle.dump(self.sph, open(file, 'wb'))
            print('pre-process sph done!')
        else:
            self.sph = pickle.load(open(file, 'rb'))
            print('load sph done!')

    def process(self, index):
        data = self.dataset[index]
        N = data.x.shape[0]
        adj = torch.zeros([N, N])
        adj[data.edge_index[0, :], data.edge_index[1, :]] = True
        sp, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
        self.sph.append(sp)
        """
        self.sph.append(adj_pinv(data, topk=10)) # ectd
        mat = cosine_similarity(data.x, data.x)
        mat = 1 / mat
        self.sph.append(mat) # cosine similarity
        """


##########################################################################
class custom_InMemoryDataset(InMemoryDataset):
    def __init__(self, df=None, df_file=None, smiles='smiles',properties= ['A','B'] ,root='temp_datasets', smiles2graph=smiles2graph,
                 transform=None, pre_transform=None, force_reload=True):
        self.df_file = df_file
        self.df = df
        if self.df_file: 
            self.df = pd.read_csv(self.df_file)
            df.reset_index(drop=True, inplace=True)
        assert isinstance(self.df,pd.core.frame.DataFrame), 'provide pandas dataframe'
        assert smiles in self.df.columns , f'{smiles} is not in the columns of {self.df}'
        self.smiles= smiles
        self.properties=properties
        for pro in self.properties:
            assert pro in self.df.columns , f'{pro} is not in the columns of df'
    
        #self.root = root
        self.root= osp.join(root)
        self.smiles2graph = smiles2graph
        super().__init__(root=self.root,  transform = transform, pre_transform=pre_transform)
        """
       
        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        dataset = custom_InMemoryDataset()
        print(dataset)
        print(dataset.data.edge_index)
        print(dataset.data.edge_index.shape)
        print(dataset.data.x.shape)
        print(dataset[100])
        print(dataset.get_idx_split())
        """
        #if self.df_file: df = pd.read_csv(self.df_file) 
        #assert isinstance(df,pd.core.frame.DataFrame), 'provide pandas dataframe'
        
        #self.folder = osp.join(root, 'peptides-structural')
        """
        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == 'y':
                shutil.rmtree(self.folder)
        """
        #self.data, self.slices = torch.load(self.processed_paths[0])
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return df_file

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def process(self):
        smiles_list = self.df[self.smiles]

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            try:            
                data = Data()
                smiles = smiles_list[i]
                graph = self.smiles2graph(smiles)
                assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                assert (len(graph['node_feat']) == graph['num_nodes'])
    
                data.__num_nodes__ = int(graph['num_nodes'])
                data.edge_index = torch.from_numpy(graph['edge_index']).to(
                    torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
                    torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                #data.y = torch.Tensor([y])
                #y = data_df.iloc[i][properties]
                for pro in self.properties: data[pro]=self.df.iloc[i][pro]
                ########
                """
                N = data.x.shape[0]
                adj = torch.zeros([N, N])
                adj[data.edge_index[0, :], data.edge_index[1, :]] = True
                data.sph, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
                """
                ############
                data=self.pre_transform(data)
                data_list.append(data)
            except:
                print('escape the following smiles which can not be convert into a graph', smiles)

        #if self.pre_transform is not None:
        #    print('running  pre_transformation .. ')
        #    data_list = [self.pre_transform(data) for data in data_list]
            
        print('Saving...')
        #data, slices = self.collate(data_list)
        #torch.save((data, slices), self.processed_paths[0])
        self.save(data_list, self.processed_paths[0])


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


###########################
class NewDataset(Dataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.dataset = original_dataset
        self.sph = []

    def __getitem__(self, index):
        data = self.dataset[index]
        data['sph'] = self.sph[index]
        return data

    def __len__(self):
        return len(self.dataset)

    def process(self, index):
        data = self.dataset[index]
        N = data.x.shape[0]
        adj = torch.zeros([N, N])
        adj[data.edge_index[0, :], data.edge_index[1, :]] = True
        sp, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
        self.sph.append(sp)
        """
        self.sph.append(adj_pinv(data, topk=10)) # ectd
        mat = cosine_similarity(data.x, data.x)
        mat = 1 / mat
        self.sph.append(mat) # cosine similarity
        """

from utils import process_sph
from torch.utils.data import  DataLoader

def load_data(args):
    if args.dataset in ['NCI1', 'NCI109', 'Mutagenicity', 'PTC_MR', 'AIDS', 'IMDB-BINARY', 'IMDB-MULTI', 'COLLAB',
                        'PROTEINS', 'DD', 'MUTAG', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K',
                        'REDDIT-MULTI-12K']:
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_tudataset(args)
    elif args.dataset[:4] == 'ogbg':
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_ogbg(args)
    elif args.dataset == 'ZINC':
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_zinc(args)
    elif args.dataset in ['CLUSTER', 'PATTERN']:
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_node_cls(args)
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, collate_fn=fn)
    val_loader = DataLoader(validation_set, batch_size=args.eval_batch_size, collate_fn=fn)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, collate_fn=fn)
    return train_loader, val_loader, test_loader, num_tasks, num_features, edge_features


def load_tudataset(args):
    transform = T.AddRandomWalkPE(walk_length=args.pe_origin_dim, attr_name='pe')
    dataset = TUDataset(os.path.join(args.data_root, args.dataset),
                        name=args.dataset,
                        pre_transform=transform
                        )
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)
    num_tasks = dataset.num_classes
    num_features = dataset.num_features
    num_edge_features = 1
    data = NewDataset(dataset)
    process_sph(args, data)
    num_training = int(len(data) * 0.8)
    num_val = int(len(data) * 0.1)
    num_test = len(data) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(data, [num_training, num_val, num_test])
    return num_tasks, num_features, num_edge_features, training_set, validation_set, test_set


def load_ogbg(args):
    if args.dataset not in ['ogbg-ppa', 'ogbg-code2']:
        transform = T.AddRandomWalkPE(walk_length=args.pe_origin_dim, attr_name='pe')
    else:
        transform = None
    dataset = PygGraphPropPredDataset(name=args.dataset, root=os.path.join(args.data_root, args.dataset),
                                      pre_transform=transform)
    num_tasks = dataset.num_tasks
    num_features = dataset.num_features
    num_edge_features = dataset.num_edge_features
    split_idx = dataset.get_idx_split()
    training_data = dataset[split_idx['train']]
    validation_data = dataset[split_idx['valid']]
    test_data = dataset[split_idx['test']]
    training_set = NewDataset(training_data)
    validation_set = NewDataset(validation_data)
    test_set = NewDataset(test_data)
    process_sph(args, training_set, 'train')
    process_sph(args, validation_set, 'val')
    process_sph(args, test_set, 'test')
    return num_tasks, num_features, num_edge_features, training_set, validation_set, test_set


def load_zinc(args):
    transform = T.AddRandomWalkPE(walk_length=args.pe_origin_dim, attr_name='pe')
    training_data = ZINC(os.path.join(args.data_root, args.dataset), split='train', subset=True,
                         pre_transform=transform)
    validation_data = ZINC(os.path.join(args.data_root, args.dataset), split='val', subset=True,
                           pre_transform=transform)
    test_data = ZINC(os.path.join(args.data_root, args.dataset), split='test', subset=True,
                     pre_transform=transform)
    training_set = NewDataset(training_data)
    validation_set = NewDataset(validation_data)
    test_set = NewDataset(test_data)
    process_sph(args, training_set, 'train')
    process_sph(args, validation_set, 'val')
    process_sph(args, test_set, 'test')
    num_tasks = 1
    num_features = 28
    num_edge_features = 4
    return num_tasks, num_features, num_edge_features, training_set, validation_set, test_set



class PeptidesStructuralDataset(InMemoryDataset):
    def __init__(self, root='datasets', smiles2graph=smiles2graph,
                 transform=None, pre_transform=None):
        """
        PyG dataset of 15,535 small peptides represented as their molecular
        graph (SMILES) with 11 regression targets derived from the peptide's
        3D structure.

        The original amino acid sequence representation is provided in
        'peptide_seq' and the distance between atoms in 'self_dist_matrix' field
        of the dataset file, but not used here as any part of the input.

        The 11 regression targets were precomputed from molecule XYZ:
            Inertia_mass_[a-c]: The principal component of the inertia of the
                mass, with some normalizations. Sorted
            Inertia_valence_[a-c]: The principal component of the inertia of the
                Hydrogen atoms. This is basically a measure of the 3D
                distribution of hydrogens. Sorted
            length_[a-c]: The length around the 3 main geometric axis of
                the 3D objects (without considering atom types). Sorted
            Spherocity: SpherocityIndex descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcSpherocityIndex
            Plane_best_fit: Plane of best fit (PBF) descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcPBF
        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'peptides-structural')

        self.url = 'https://www.dropbox.com/s/464u3303eu2u4zp/peptide_structure_dataset.csv.gz?dl=1'
        self.version = '9786061a34298a0684150f2e4ff13f47'  # MD5 hash of the intended dataset file
        self.url_stratified_split = 'https://www.dropbox.com/s/9dfifzft1hqgow6/splits_random_stratified_peptide_structure.pickle?dl=1'
        self.md5sum_stratified_split = '5a0114bdadc80b94fc7ae974f13ef061'

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == 'y':
                shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'peptide_structure_dataset.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), 'w').close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print('Stop download.')
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir,
                                       'peptide_structure_dataset.csv.gz'))
        smiles_list = data_df['smiles']
        target_names = ['Inertia_mass_a', 'Inertia_mass_b', 'Inertia_mass_c',
                        'Inertia_valence_a', 'Inertia_valence_b',
                        'Inertia_valence_c', 'length_a', 'length_b', 'length_c',
                        'Spherocity', 'Plane_best_fit']
        # Normalize to zero mean and unit standard deviation.
        data_df.loc[:, target_names] = data_df.loc[:, target_names].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0)

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][target_names]
            graph = self.smiles2graph(smiles)

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(
                torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
                torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([y])

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """ Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(self.root,
                              "splits_random_stratified_peptide_structure.pickle")
        with open(split_file, 'rb') as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict

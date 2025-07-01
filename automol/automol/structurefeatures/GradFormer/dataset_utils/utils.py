from __future__ import annotations

import os , csv, pickle
from tqdm import tqdm
import networkx as nx
from torch_geometric.utils import to_networkx
from typing import Tuple, Dict, List
from torch_geometric.data import Data
import torch
from torch.nn import LeakyReLU, ReLU
import torch.nn.functional as F

import numpy as np
import torch, os ,sys
import time
from tqdm import tqdm
import numpy as np
import subprocess, psutil
import pylab as pl
import pandas as pd

###





def set_device(with_cuda=True, cuda_devices= None):
        '''
        cuda_devices should be list of GPU IDs to use such as "0,1,2,3,4,5,6,7"
        cuda_devices= all use all gpus
        '''
        if not torch.cuda.is_available():
            print('Gpus are not avialable....Using Cpu as advice')
            device =torch.device('cpu')
            return device , None
        if torch.cuda.is_available() and with_cuda:
            if (not cuda_devices) or (torch.cuda.device_count()==1) :
                print("Using one GPU")
                device = torch.device("cuda:0")
                return device , [0]
            elif cuda_devices:
                if cuda_devices =='all':
                    print("Using all the avialable %d GPUS" % torch.cuda.device_count())
                    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    device_ids=[d for d in range(torch.cuda.device_count())]
                else:
                    device_ids=[int(d) for d in cuda_devices.split(',')]
                    NRgpus= len(device_ids)
                    assert NRgpus >0 and NRgpus <= torch.cuda.device_count(), 'Number of gpus shoulb be in the range 0- total Gpus'
                device =torch.device(f'cuda:{device_ids[0]}')
                print('Using cuda with the following master device device', device)
                #torch.cuda.set_device(device)
                print('devices list:',device_ids )
                return device, device_ids



def get_model_for_prediction(model_file=None,use_gpu=True):
    """
    get a pretrained model ready for prediction

    model_file: loacal .pt file of the model
    """

    print(f'loading the model from file {model_file}')
    assert os.path.isfile(model_file),'model file is not found!'
    if use_gpu and  torch.cuda.is_available():
        device = torch.device(f'cuda:0')
    else:
        device =torch.device('cpu')
    print('using device:',device)
    checkpoint = torch.load(model_file , map_location='cpu')
    model= checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    model=model.to(device)
    return model

def process_hop(sph, gamma, hop, slope=0.1):
    leakyReLU = LeakyReLU(negative_slope=slope)
    sph = sph.unsqueeze(1)
    sph = sph - hop
    sph = leakyReLU(sph)
    sp = torch.pow(gamma, sph)
    return sp


def process_sph(args, data, split=None):
    os.makedirs(f'./sph', exist_ok=True)
    if split is None:
        file = f'./sph/{args.dataset}.pkl'
    else:
        file = f'./sph/{args.dataset}_{split}.pkl'
    if not os.path.exists(file):
        print('pre-process start!')
        progress_bar = tqdm(desc='pre-processing Data', total=len(data), ncols=70)
        for i in range(len(data)):
            data.process(i)
            progress_bar.update(1)
        progress_bar.close()
        pickle.dump(data.sph, open(file, 'wb'))
        print('pre-process down!')
    else:
        data.sph = pickle.load(open(file, 'rb'))
        print('load sph down!')




def floyd_warshall_source_to_all(G, source, cutoff=None):
    if source not in G:
        raise nx.NodeNotFound("Source {} not in G".format(source))

    edges = {edge: i for i, edge in enumerate(G.edges())}

    level = 0  # the current level
    nextlevel = {source: 1}  # list of nodes to check at next level
    node_paths = {source: [source]}  # paths dictionary  (paths to key from source)
    edge_paths = {source: []}

    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in G[v]:
                if w not in node_paths:
                    node_paths[w] = node_paths[v] + [w]
                    edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
                    nextlevel[w] = 1

        level = level + 1

        if cutoff is not None and cutoff <= level:
            break

    return node_paths, edge_paths


def all_pairs_shortest_path(G) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
    node_paths = {n: paths[n][0] for n in paths}
    edge_paths = {n: paths[n][1] for n in paths}
    return node_paths, edge_paths


def shortest_path_distance(data: Data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    G = to_networkx(data)
    node_paths, edge_paths = all_pairs_shortest_path(G)
    return node_paths, edge_paths


def num_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0.0:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(torch.nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


#################################











##############################
def results_to_file(args, val, test):

    if not os.path.exists('./results/{}'.format(args.dataset)):
        print("=" * 20)
        print("Create Results File !!!")

        os.makedirs('./results/{}'.format(args.dataset))

    filename = "./results/{}/result.csv".format(
        args.dataset)

    headerList = ["Method", "Layer-Num", "Slope", "n_hop", "gamma", "drop_out", "attn_drop", "drop_path",
                  "::::::::", "val", "test"]

    with open(filename, "a+") as f:

        f.seek(0)
        header = f.read(6)
        if header != "Method":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()

        line = "{}, {}, {}, {}, {}, {}, {}, {}, :::::::::, {:.5f}, {:.5f}\n".format(
            args.model_type, args.num_layers, args.slope, args.n_hop, args.gamma, args.dropout,
            args.attn_dropout, args.drop_prob, val, test)
        f.write(line)
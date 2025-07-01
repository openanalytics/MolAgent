"""implementation llm dataset for transformer w smiles.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""

import re
from collections import Counter
import random
import numpy as np , pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class transformerDataset(Dataset):
    def __init__(self, df, vocab, seq_len=220, transform=None,tgt_transform=None,smiles='smiles', properties= []):
        assert isinstance(df,pd.core.frame.DataFrame), 'provide pandas dataframe'
        self.df = df.reset_index(drop=True)
        self.smiles= smiles
        assert smiles in df.columns
        self.properties=properties
        if len(properties):
            for pro in self.properties:
                assert pro in df.columns , f'{pro} is not in the columns of {self.df}'
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = transform
        self.tgt_transform=tgt_transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        tt= self.df.iloc[item]
        src_sm = tt[self.smiles]
        if self.transform:
            src_sm = self.transform(src_sm)
        src_int=self.vocab.smile2int( src_sm, max_smile_len=self.seq_len, with_eos=True, with_sos=True, return_len=False)
        tgt_sm=tt[self.smiles]
        if self.tgt_transform:
            tgt_sm= self.tgt_transform(tgt_sm)
        tgt_int=self.vocab.smile2int( tgt_sm, max_smile_len=self.seq_len, with_eos=True, with_sos=True, return_len=False)
        #for i, sentence in enumerate(batch):
        #masks.append([False for _ in range(len(sentence))] + [True for _ in range(seq_length - len(sentence))])
        #mask=[ True if s == self.vocab.pad_index else False for s in X ] #(X== self.vocab.pad_index)
        output = { "src":src_int  , "tgt":tgt_int } #, "properties":tt[self.properties].tolist() }
        if len(self.properties):
            for pro in self.properties:       output[pro]=tt[pro]
        return {key: torch.tensor(value) for key, value in output.items()}

########################################################################
class Vocab(object):
    '''
    Smile tockinizer
    '''
    def __init__(self, smile_lists=None,specials=["<pad>", "<unk>",
                                                  "<eos>",
                                                  "<sos>",
                                                  "<sof>", # start of fragment
                                                  "<eof>",# end of fragment
                                                  "<mask>","<cls>" ,
                                                  "[*]","[*:0]","[*:1]","[*:2]","[*:3]","[*:4]","[*:5]",
                                                  ".","[Fm]","[Es]", ">>",  "[Dummy]"], min_freq=1):
        from collections import Counter
        self.specials=specials
        self.min_freq = max(min_freq, 1)
        self.tok_list = []
        self.dict_tok2int =dict()
        self.freq= Counter()
        if self.specials:
            self.tok_list =list(self.specials)
            self.dict_tok2int = {tok: i for i, tok in enumerate(self.tok_list)}
            self.pad_index  = self.dict_tok2int["<pad>"]  # 0
            self.unk_index  = self.dict_tok2int["<unk>"]  # 1
            self.eos_index  = self.dict_tok2int["<eos>"]  # 2
            self.sos_index  = self.dict_tok2int["<sos>"]  # 3
            self.mask_index = self.dict_tok2int["<mask>"] # 4
            self.eof_index  = self.dict_tok2int["<eof>"]
            self.sof_index  = self.dict_tok2int["<sof>"]
            self.cls_index  = self.dict_tok2int["<cls>"]
            self.sep_index  = self.dict_tok2int["."]
            self.fragment_index  = self.dict_tok2int["[Fm]"]
            self.fragment_index2  = self.dict_tok2int["[Es]"]
            self.dummy_index = self.dict_tok2int["[Dummy]"]
        if smile_lists is not None:
            self.add_to_vocab(smile_lists)
    def add_to_vocab(self,smile_lists, min_freq=None):
        if not min_freq:
            min_freq=self.min_freq
        print(f"Building Vocab and adding new token with freq >= {min_freq}...")
        new_tokens= self.get_tokens(smile_lists)
        for t in new_tokens:
            self.freq[t] += new_tokens[t]
            if t not in self.tok_list and new_tokens[t] >= min_freq:
                self.tok_list.append(t)
                self.dict_tok2int[t] = len(self.tok_list) - 1

    def get_tokens(self, smile_lists):
        """
        get the unique smile letters from a list of smiles
        return a dic of the tokens with their frequency
        """
        out= Counter()
        #out=[]
        for sm in smile_lists:
            for s in self.smile_split(sm):
                out[s] += 1
            #new=set(self.smile_split(sm))
            #out += [ t for t in new if t not in out]
        return out
    def smile_split(self,smi):
        if smi is None: return []
        pattern =  "(\[[^\]]+]|Br?|Cl?|Al|As|Ag|Au|Be|Ba|Bi|Ca|Cu|Fe|Kr|He|Li|Mg|Mn|Na|Ni|Ra|Rb|Si|si|se|Se|Sr|Te|te|Xe|Zn|>>|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        assert smi == ''.join(tokens) , print(smi)
        return tokens#' | '.join(tokens)

    def smile2int(self, smile, max_smile_len=None, with_eos=False, with_sos=False, return_len=False):
        ll =self.smile_split(smile)
        seq = [self.dict_tok2int.get(s, self.unk_index) for s in ll]
        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq
        origin_seq_len = len(seq)
        if max_smile_len is None:
            pass
        elif len(seq) <= max_smile_len:
            seq += [self.pad_index for _ in range(max_smile_len - len(seq))]
        else:
            seq = seq[:max_smile_len]
        return (seq, origin_seq_len) if return_len else seq

    def int2smiles(self, seq):
        smi=[]
        for idx in seq:
            if idx == self.eos_index:
                break
            elif idx in [self.sos_index, self.pad_index, self.cls_index ]:
                continue
            elif idx in [self.mask_index, self.unk_index]:
                s= "{}".format(self.tok_list[idx])
            else:
                s =self.tok_list[idx]
            smi.append(s)
        return "".join(smi)
    def __len__(self):
        return len(self.tok_list)

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        import pickle
        print(f"Loading Vocabulary from file {vocab_path}...")
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        import pickle
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

    def get_frequency(self):
        ''' show the frequeny of the tokens'''
        return self.freq.most_common()

    def extend_tokens_wrt_freq(self ,min_freq=1 ):
        ''' extend the list of the tokens to
            include more tokens having the min freq
        '''
        assert (min_freq > 0 and min_freq < self.min_freq), 'invalid min_freq'
        self.min_freq =min_freq
        for t in self.freq:
            if t not in self.tok_list and self.freq[t] >= self.min_freq:
                    self.tok_list.append(t)
                    self.dict_tok2int[t] = len(self.tok_list) - 1
    def __eq__(self, other):
        '''compare two vocabularies'''
        if self.dict_tok2int == other.dict_tok2int:
            return True
        else:
            return False


class smile2canonical():
    """returns the RDKit canonical SMILES for a input SMILES sequnce."""
    def __init__(self):
        pass
    def __call__(self, sml):
        try:
            m = Chem.MolFromSmiles(sml)
        except:
            return float('nan')
        if m is not None:
            return Chem.MolToSmiles(m, canonical=True)


class randomize_smile():
    """randomizes a SMILES sequnce. This was adapted from the
    implemetation of E. Bjerrum 2017, SMILES Enumeration as Data Augmentation
    for Neural Network Modeling of Molecules.
    return nan if SMILES is not interpretable.
    """
    def __init__(self):
        pass
    def __call__(self, sml):
        try:
            m = Chem.MolFromSmiles(sml)
            ans = list(range(m.GetNumAtoms()))
            np.random.shuffle(ans)
            nm = Chem.RenumberAtoms(m, ans)
            return Chem.MolToSmiles(nm, canonical=False)
        except:
            return sml

class custom_randomize_smile():
    """
    10% keep org , 10% rdkit canon , 80% randomize
    """
    def __init__(self):
        pass
    def __call__(self, sml):
        prob = random.random()
        if prob <= 0.10:
                return sml ## chemaxon
        try:        m = Chem.MolFromSmiles(sml)
        except:         return sml
        if prob <= 0.20:
            return Chem.MolToSmiles(m, canonical=True) # rdkit canon
        else:
            try:
                ans = list(range(m.GetNumAtoms()))
                np.random.shuffle(ans)
                nm = Chem.RenumberAtoms(m, ans)
                return Chem.MolToSmiles(nm, canonical=False)
            except: return sml

class fragment_custom_randomize_smile():
    """
    10% keep org , 10% rdkit canon , 80% randomize
    """
    def __init__(self):
        pass
    def __call__(self, sml):
        prob = random.random()
        if prob <= 0.10:
                return sml ## chemaxon
        try:        m = Chem.MolFromSmiles(sml,sanitize=False)
        except:         return sml
        if prob <= 0.20:
            return Chem.MolToSmiles(m, canonical=True) # rdkit canon
        else:
            try:
                ans = list(range(m.GetNumAtoms()))
                np.random.shuffle(ans)
                nm = Chem.RenumberAtoms(m, ans)
                return Chem.MolToSmiles(nm, canonical=False)
            except: return sml
class fragment_smile2canonical():
    """returns the RDKit canonical SMILES for a input SMILES sequnce."""
    def __init__(self):
        pass
    def __call__(self, sml):
        try:
            m = Chem.MolFromSmiles(sml,sanitize=False)
        except:
            return float('nan')
        if m is not None:
            return Chem.MolToSmiles(m, canonical=True)



class compose_pipeline(object):
    """Composes a pipeline of several transforms on the input.
    to be passed to torch.utils.data.Dataset  as an argument transform=transforms.compose_pipeline([....
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, smi):
        for t in self.transforms:
            smi = t(smi)
        return smi

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

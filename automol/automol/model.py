"""The transformer model using smiles.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""
#from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Type
import torch
import sys ,os
import math
import numpy as np
from torch import nn , Tensor
from torch.nn import functional as F
from pkg_resources import resource_filename

#######################################################################
class transformer_init():
    def __init__(
        self,
        vocab=None,
        hidden_size=250,
        num_hidden_layers=5,
        num_attention_heads=5,
        hidden_activation="gelu",# "relu"
        hidden_dropout=0.1,
        attention_dropout=0.1,
        max_len=220,
        max_num_of_segments=10,
        initializer_range=0.02,
        norm_eps=1e-12,
        learn_position_embeddings=True,
        key_padding_mask=True
    ):
        self.vocab=None
        self.pad_index=0
        self.vocab_size =50
        if vocab:
            self.vocab= vocab
            self.pad_index= vocab.pad_index
            self.vocab_size = len(vocab)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_activation = hidden_activation
        self.hidden_dropout= hidden_dropout
        self.attention_dropout= attention_dropout
        self.max_len = max_len
        self.max_num_of_segments = max_num_of_segments
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.learn_position_embeddings=learn_position_embeddings
        self.key_padding_mask=key_padding_mask
    @staticmethod
    def load_config(f_path) :
        import pickle
        print(f"Loading config file from file {f_path}...")
        with open(f_path, "rb") as f:
            return pickle.load(f)

    def save_config(self, f_path):
        import pickle
        print(f"Saving config file to file {f_path}...")
        with open(f_path, "wb") as f:
            pickle.dump(self, f)
    def model_init(self, model_class=None,device=None ,device_ids=None ):
        if model_class:
            model =model_class(self)
            print('Total parameters:', sum(p.numel() for p in model.parameters()))
        if device:
            model=model.to(device)
            if device_ids and len(device_ids) > 1:
                print("Using", len(device_ids) , "GPUs for DataParallel")
                model = nn.DataParallel(model, device_ids=device_ids, dim=1)
        return model
    def init_pretrained_model(self,PATH,  model_class=None,device='cpu' ,device_ids=None ):
        model =model_class(self)
        print(f'loading parameters from file :{PATH}')
        model.load_state_dict(torch.load(PATH, map_location=device))
        print('Total parameters:', sum(p.numel() for p in model.parameters()))
        if device:
            model=model.to(device)
        if device_ids and len(device_ids) > 1:
            print("Using", len(device_ids) , "GPUs for DataParallel")
            model = nn.DataParallel(model, device_ids=device_ids, dim=1)
        return model

#######################################################################
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_size):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, hidden_size).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, src):
        return self.pe[:, :src.size(1)]
#######################################################################
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_size, padding_idx=0):
        super().__init__()
        self.max_len=max_len
        self.position_embeddings= nn.Embedding(self.max_len, hidden_size, padding_idx=padding_idx)
    def forward(self, src):
        input_shape = src.size()
        seq_len=input_shape[0]
        device = src.device
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(1).expand(input_shape)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings
#######################################################################
class TransformerEmbeddings(nn.Module):
    """
       embeddings= positional, token embeddings
       positional is learned embeddings or Sinusoidal
       provided num_embeddings and padding_idx override the config ones
    """
    def __init__(self, config, num_embeddings=None, padding_idx=None):
        super().__init__()
        if not num_embeddings:
            num_embeddings=config.vocab_size
        if not padding_idx:
            padding_idx=config.pad_index
        self.word_embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=config.hidden_size, padding_idx=padding_idx)
        if config.learn_position_embeddings:
            self.position_embeddings = LearnedPositionalEmbedding(config.max_len, config.hidden_size, padding_idx=0)
        else:
             self.position_embeddings = SinusoidalPositionalEmbedding(config.max_len, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, src=None, Segment_ids=None):
        inputs_embeds = self.word_embeddings(src)
        position_embeddings = self.position_embeddings(src)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
#######################################################################
class MultiheadAttention(nn.Module):
    """
    returns:
            out['output']          tgt_len, bsz, embed_dim
            out['attentions']      tgt_len, bsz,num_heads, src_len     optional return_attentions=False,
            out['heads_outputs']   tgt_len, bsz,num_heads, head_dim    optional return_heads_output=False
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert  embed_dim % num_heads == 0 , "embed_dim must be divisible by num_heads"
        self.head_dim=  embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.pooling_heads = nn.Linear(embed_dim, embed_dim)
    def forward(self, query, key, value,
                key_padding_mask=None,
                 attn_mask=None,
                 return_attentions=False,
                 return_heads_output=False):
        """
        query: (tgt_len, bsz, embed_dim)
        key:   (src_len, bsz, embed_dim)
        value: (src_len, bsz, embed_dim)
        key_padding_mask: (bsz, src_len) where N is the batch size, S is the source sequence length. If a ByteTensor is provided,
                        the non-zero positions will be ignored while the position with the zero positions will be
                        unchanged. If a BoolTensor is provided, the positions with the value of True will be ignored while the position
                        with the value of False will be unchanged.
        attn_mask: 2D mask (tgt_len, src_len)
        """
        tgt_len, bsz, embed_dim = query.size()
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)
        self.scaling = torch.sqrt(torch.FloatTensor([self.head_dim])).to(key.device)
        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
                attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
                    'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                    warnings.warn("Use bool tensor instead of Byte tensor for attn_mask in  MultiheadAttention")
                    attn_mask = attn_mask.to(torch.bool)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.
        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn("Use bool tensor instead of Byte tensor for key_padding_mask in MultiheadAttention")
            key_padding_mask = key_padding_mask.to(torch.bool)
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.k_linear(value)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # q  bsz * num_heads , tgt_len, head_dim
        # k  bsz * num_heads , src_len, head_dim
        src_len = k.size(1)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))/self.scaling
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            #The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        attn_output = torch.bmm(attn_output_weights, v)# bsz * self.num_heads, tgt_len, self.head_dim
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        output = self.pooling_heads(attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)) # tgt_len, bsz, embed_dim
        out = {'output':output}
        if return_attentions:
            #attn_output_weights   bsz * num_heads, tgt_len, src_len
            attn_output_weights = attn_output_weights.contiguous().view(bsz, self.num_heads, tgt_len, src_len).permute(2, 0,1,3).contiguous()
            #attn_output_weights  tgt_len, bsz,num_heads, src_len
            out['attentions'] = attn_output_weights
        if return_heads_output:
            attn_output = attn_output.transpose(0, 1).view(tgt_len, bsz, self.num_heads, self.head_dim)
            # tgt_len, bsz,num_heads, head_dim
            out['heads_outputs'] = attn_output
        return out # output (tgt_len, bsz, embed_dim), attn_output_weights ( tgt_len, bsz,num_heads, src_len),return_heads_output(tgt_len, bsz,num_heads, head_dim)
##################################################################################
class TransEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=1000, dropout=0.1, activation="gelu"):
        super(TransEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if activation == "relu":
            self.activation= F.relu
        elif activation == "gelu":
            self.activation= F.gelu
        else:
            raise RuntimeError(f"activation should be relu/gelu, not {activation}")
    ############
    def forward(self, src: Tensor,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                       return_attentions=False,
                     return_heads_output=False) -> Tensor:
        """Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        mha = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,
                            return_attentions=return_attentions,
                 return_heads_output=return_heads_output)

        src2= mha['output']
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        out={}
        out['output'] =src
        if return_attentions: out['attentions'] =mha['attentions']
        if return_heads_output :out['heads_outputs']=mha['heads_outputs']
        return out
#######################################################################
import copy
class TransEncoder(nn.Module):

    def __init__(self, d_model=250,
                 nhead=10,
                 num_encoder_layers=5,
                 dim_feedforward=1000,
                 dropout=0.1,
                 activation="gelu" ):
        super(TransEncoder, self).__init__()
        encoder_layer = TransEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        #self.encoder_norm = nn.LayerNorm(d_model)
        self.encoderlyers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_encoder_layers)])
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers=num_encoder_layers
    def forward(self, src,
                       src_key_padding_mask =None,
                        src_mask= None,
                        return_output_layers=[],
                       return_atten_layers=[],
                       return_heads_layers=[]):
        '''
        '''
        if not len(return_output_layers):
            return_output_layers=[self.num_layers-1]
        assert src.size(2) == self.d_model,"the feature number of must be equal to d_model"
        hidden_states=src
        out={}
        for i, layer in enumerate(self.encoderlyers):
            return_attentions=False
            return_heads_output=False
            return_output=False
            if i in return_atten_layers:  return_attentions=True
            if i in return_heads_layers:  return_heads_output=True
            if i in return_output_layers: return_output=True
            layer_outputs = layer(hidden_states,
                                  src_key_padding_mask=src_key_padding_mask,
                                  src_mask=src_mask,
                                  return_attentions=return_attentions,
                                  return_heads_output=return_heads_output)
            hidden_states = layer_outputs['output']
            lout={}
            if return_attentions:   lout['attentions']= layer_outputs['attentions']
            if return_heads_output: lout['heads_outputs']= layer_outputs['heads_outputs']
            if return_output:       lout['output']= layer_outputs['output']
            if len(lout):
                out[f'{i}']= lout
        '''##  last layer
        if f'{self.num_layers-1}' not in out:
            out[f'{self.num_layers-1}']= layer_outputs'''
        return out
    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

################################################################################
class Smiles_Encoder(nn.Module):
    '''
    '''
    def __init__(self, config ):
        '''
        '''
        super(Smiles_Encoder, self).__init__()
        self.pad_index= config.pad_index
        self.Embeddings = TransformerEmbeddings(config)
        self.encoder= TransEncoder(d_model=config.hidden_size,
                                             nhead=config.num_attention_heads,
                                             num_encoder_layers=config.num_hidden_layers,
                                             dim_feedforward=config.hidden_size*4,
                                             dropout=config.hidden_dropout,
                                             activation=config.hidden_activation)
    def forward(self, src, src_key_padding_mask=None,
                return_output_layers=[],
               return_atten_layers=[],
                       return_heads_layers=[]):
        # src: (T,B)
        src_embedded = self.Embeddings(src)  # (T,B,H)
        device = src.device
        if not src_key_padding_mask:
            src_key_padding_mask = torch.t(src).eq(self.pad_index).to(device) #(B,T)
        encoder_out= self.encoder(  src_embedded,
                                    src_key_padding_mask=src_key_padding_mask,
                                  return_output_layers=return_output_layers,
                                    return_atten_layers=return_atten_layers,
                                    return_heads_layers=return_heads_layers)
        #memory=encoder_out['output']
        #if len(return_atten_layers)  or len(return_heads_layers):
        return encoder_out
#######################################################################
class TransDecoderLayer(nn.Module):
    """
    """
    def __init__(self, d_model, nhead, dim_feedforward=1000, dropout=0.1, activation="gelu"):
        super(TransDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        if activation == "relu":
            self.activation= F.relu
        elif activation == "gelu":
            self.activation= F.gelu
        else:
            raise RuntimeError(f"activation should be relu/gelu, not {activation}")
    def forward(self, tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                return_attentions=False,
                return_heads_output=False) -> Tensor:

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                             return_attentions=False,
                             return_heads_output=False)['output']
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        mha = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                  return_attentions=return_attentions,
                                  return_heads_output=return_heads_output)

        tgt = tgt + self.dropout2(mha['output'])
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        out={}
        out['output'] =tgt
        if return_attentions: out['attentions'] =mha['attentions']
        if return_heads_output :out['heads_outputs']=mha['heads_outputs']
        return out


################################################################################
class TransDecoder(nn.Module):
    """
    has_mask: mask the following tokens
    """
    def __init__(self, d_model            = 250,
                       nhead              = 10,
                       num_decoder_layers = 5,
                       dim_feedforward    = 1000,
                       dropout            = 0.1,
                       activation         = "gelu" ):
        super(TransDecoder, self).__init__()
        decoder_layer = TransDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.decoderlyers  =nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_decoder_layers)])
        self._reset_parameters()
        self.d_model  = d_model
        self.nhead    = nhead
        self.num_layers=num_decoder_layers
    def forward(self, tgt, memory,
                            tgt_key_padding_mask=None,
                            tgt_mask=None,
                            memory_key_padding_mask=None,
                            return_output_layers=[],
                            return_atten_layers=[],
                            return_heads_layers=[]):
        if not len(return_output_layers):
            return_output_layers=[self.num_layers-1]
        assert tgt.size(2) == self.d_model,"the feature number of  tgt must be equal to d_model"
        '''device = tgt.device
        if self.has_mask:
            tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(device)'''
        hidden_states=tgt
        out={}
        for i, layer in enumerate(self.decoderlyers):
            return_attentions=False
            return_heads_output=False
            return_output=False
            if i in return_atten_layers:  return_attentions=True
            if i in return_heads_layers:  return_heads_output=True
            if i in return_output_layers: return_output=True
            layer_outputs = layer(hidden_states, memory,
                                  tgt_mask=tgt_mask,
                                     memory_mask=None,
                                     tgt_key_padding_mask=tgt_key_padding_mask,
                                     memory_key_padding_mask=memory_key_padding_mask,
                                      return_attentions=return_attentions,
                                      return_heads_output=return_heads_output)
            hidden_states = layer_outputs['output']
            lout={}
            if return_attentions:   lout['attentions']= layer_outputs['attentions']
            if return_heads_output: lout['heads_outputs']= layer_outputs['heads_outputs']
            if return_output:       lout['output']= layer_outputs['output']
            if len(lout):
                out[f'{i}']= lout
        return out
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
           Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
###########################################################
class seq_Generator(nn.Module):
    """
    output: the log_softmax of the vocabulary
    """
    def __init__(self, config):
        super(seq_Generator, self).__init__()
        self.task_name='seq_Generator'
        self.n_output=config.vocab_size
        self.pad_index=config.pad_index
        self.decoder= TransDecoder( d_model=config.hidden_size,
                                              nhead=config.num_attention_heads,
                                              num_decoder_layers=config.num_hidden_layers,
                                              dim_feedforward=config.hidden_size*4,
                                              dropout=config.hidden_dropout,
                                               activation=config.hidden_activation )
        self.seq_out = nn.Linear(config.hidden_size, config.vocab_size)
        self.uncertainty  = torch.nn.Parameter(torch.zeros(1))
    def forward(self, tgt_embedded, memory ,tgt_key_padding_mask=None,tgt_mask=None):
        """
        inputs :  embedings of the target ,memory from the encoder
        output: the log_softmax of the vocabulary
        """
        de_out = self.decoder(tgt=tgt_embedded, memory=memory ,tgt_key_padding_mask=tgt_key_padding_mask, tgt_mask=tgt_mask)
        last_layer=self.decoder.num_layers-1
        hidden=de_out[f'{last_layer}']['output']
        seq_out = self.seq_out(hidden) # (T,B,V)
        seq_out=  F.log_softmax(seq_out, dim=-1)
        return seq_out
    def loss_fn(self, probs, tgt):
        # probs (T,B,V)
        # tgt (T,B)
        loss_fuc=nn.NLLLoss(ignore_index=self.pad_index, reduction='none')
        probs_flatten=probs.view(-1, self.n_output)
        tgt_flatten = tgt.contiguous().view(-1)
        mask=(tgt_flatten== self.pad_index)
        #loss= loss_fuc(probs_flatten,tgt_flatten)
        loss= loss_fuc(probs_flatten,tgt_flatten)
        ## acc
        predicted_labels = torch.argmax(probs_flatten, dim=-1)
        acc=100.0* (predicted_labels[~mask] == tgt_flatten[~mask]).sum().item()/len(tgt_flatten[~mask])
        return loss ,acc



####################



##################################################
class prediction_layer(nn.Module):
    ""
    def __init__(self, d_model,
                 activation=None,leaky_relu_slope=0.01, dropout=0.1,out_dim=None ):
        super(prediction_layer, self).__init__()
        if not out_dim:
            out_dim= d_model
        self.linear=nn.Linear(d_model, out_dim)
        self.activation= activation
        if not self.activation:
            self.activation=nn.LeakyReLU(leaky_relu_slope)
        self.LayerNorm = torch.nn.LayerNorm(out_dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self, hidden):
        out = self.activation(self.linear(hidden))
        return self.dropout(self.LayerNorm(out))

###################################################
class regression_head(nn.Module):
    ""
    def __init__(self, task_name,hidden_size=250,pred_hidden_dropout=0.1,
     n_layers=1, n_output=1 ,out_dim=None ):
        super(regression_head, self).__init__()
        if not out_dim:
            out_dim= hidden_size
        self.task_name=task_name
        self.n_output=n_output
        if n_layers > 1:
            layer=prediction_layer(d_model=hidden_size, dropout=pred_hidden_dropout)
            layers=[copy.deepcopy(layer) for i in range(n_layers-1)]
        else:
            layers=[]
        las_layer=prediction_layer(d_model=hidden_size, dropout=pred_hidden_dropout, out_dim=out_dim)
        layers= layers+[las_layer]
        self.layers= nn.ModuleList(layers)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(pred_hidden_dropout)
        self.linear=nn.Linear(out_dim, n_output)
        self.uncertainty  = torch.nn.Parameter(torch.zeros(1))
    def forward(self, hidden_states):
        hidden=hidden_states[0, :].unsqueeze(0)
        for  layer in self.layers:
            hidden= layer(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return self.linear(hidden)
    def loss_fn(self, y_pred, y_true):
        loss_fuc=torch.nn.MSELoss(reduction='none')
        y_pred=y_pred.contiguous().view(-1)#, self.n_output)
        y_true = y_true.contiguous().view(-1).type(y_pred.dtype)
        # Missing data are nan's
        mask = torch.isnan(y_true)
        loss= loss_fuc(y_pred[~mask] ,y_true[~mask] )
        ## R2
        R2=1-(torch.sum((y_pred[~mask] - y_true[~mask]) ** 2) / torch.sum((y_true[~mask] - y_true[~mask].mean()) ** 2)).item()
        return loss ,R2
################################################
class Classification_head(nn.Module):
    """
    N-class classification model
    """
    def __init__(self, task_name,n_layers=1,n_output=2,hidden_size=250,
                 pred_hidden_dropout=0.1,ignore_index=-100,labels_id2name=None) :
        super().__init__()
        self.task_name=task_name
        self.n_output=n_output
        self.ignore_index=ignore_index
        self.labels_id2name=labels_id2name
        layer=prediction_layer(d_model=hidden_size, dropout=pred_hidden_dropout)
        self.layers= nn.ModuleList([copy.deepcopy(layer) for i in range(n_layers)])
        self.linear=nn.Linear(hidden_size, self.n_output)
        self.uncertainty  = torch.nn.Parameter(torch.zeros(1))
        #self.loss_fn =loss_fn
        #if not self.loss_fn:   self.loss_fn= nn.NLLLoss(ignore_index=-10000,reduction='none')
    def forward(self, hidden_states):
        hidden=hidden_states[0, :].unsqueeze(0)
        for  layer in self.layers:
            hidden= layer(hidden)
        out  = self.linear(hidden) # (T,B,n_class)
        return  F.log_softmax(out, dim=-1)
    def loss_fn(self, probs, tgt ):
        # probs (T,B,n_class)
        # tgt (T,B)
        #device= probs.device
        device=next(self.parameters()).device
        loss_fuc=nn.NLLLoss(ignore_index=self.ignore_index, reduction='none')
        probs_flatten=probs.view(-1, self.n_output)
        tgt_flatten = tgt.contiguous().view(-1)
        loss= loss_fuc(probs_flatten,tgt_flatten)#.to(device)
        ## acc
        mask=(tgt_flatten == self.ignore_index)
        predicted_labels = torch.argmax(probs_flatten, dim=-1)
        acc=100.0* (predicted_labels[~mask] == tgt_flatten[~mask]).sum().item()/len(tgt_flatten[~mask])
        return loss ,acc


######################################
def make_tasks( tasks_dict):
        '''
        example:
        inputs: tasks_dict={
                    'chemaxon_LogD' :{'is_regression':1 , 'n_output':1,'n_layers': 1,'hidden_size':250, 'out_dim':50},
                    'pic50':{'is_regression':0 , 'n_output':1,'n_layers': 1}}
        output:
        tasks list
        '''
        #if self.pool:        hidden_size=hidden_size
        tasks=[]
        for task_name in tasks_dict:
            try: pred_hidden_dropout=tasks_dict[task_name]['pred_hidden_dropout']
            except:pred_hidden_dropout=0.1
            try: out_dim=tasks_dict[task_name]['out_dim']
            except:out_dim=None
            if  tasks_dict[task_name]['is_regression']:
                model= regression_head(task_name=task_name,
                                        hidden_size=tasks_dict[task_name]['hidden_size'],
                                      n_layers=tasks_dict[task_name]['n_layers'],
                                      n_output=tasks_dict[task_name]['n_output'],
                                      pred_hidden_dropout=pred_hidden_dropout, out_dim=out_dim)
                #tasks_dict[task_name]['loss_fn']= torch.nn.MSELoss(reduction='none')
            else:#Classification_head
                try:  ignore_index=tasks_dict[task_name]['mask']
                except:ignore_index= -100
                try: labels_id2name=tasks_dict[task_name]['labels_id2name']
                except: labels_id2name=None
                model= Classification_head(task_name=task_name,
                                        hidden_size=tasks_dict[task_name]['hidden_size'],
                                      n_layers=tasks_dict[task_name]['n_layers'],
                                      n_output=tasks_dict[task_name]['n_output'],
                                      labels_id2name=labels_id2name,
                                      ignore_index=ignore_index,
                                      pred_hidden_dropout=pred_hidden_dropout)
            tasks.append(model)
        return tasks
########################################################################################

import inspect
from torch import optim
class Fine_tunning_Tasks(nn.Module):
    '''
    fine_tunning_Tasks
    '''
    def __init__(self, tasks, smiles_generator=False ,fine_tune_scale=None
                    ,lr=0.0005 , encoder= 'CHEMBL27',vocab=None, output_layer=None):
        '''
        tasks: can be a dict or tasks list
        smiles_generator: if True add a pretrained smiles generator
        tasks: list of tasks init models
        fine_tune_scale: scale the lr of the base model,None: freezing the parameters of the base model
        '''
        super(Fine_tunning_Tasks, self).__init__()
        if encoder and type(encoder) != type('aa'):
            self.Smiles_Encoder= encoder
            assert vocab, 'Provide the corr. vocab of the encoder'
            self.vocab= vocab
        else:
            self.base_model_file=resource_filename(
                'automol.trained_models',
                'CHEMBL27_ENUM_SMILES_ENCODER.pt'
            )
            assert os.path.isfile(self.base_model_file),f'provide encoder file {self.base_model_file} not found!'
            print(f'loading the base model from file: {self.base_model_file}')
            checkpoint = torch.load(self.base_model_file , map_location='cpu')
            self.Smiles_Encoder=checkpoint['Smiles_Encoder']
            self.Smiles_Encoder.load_state_dict(checkpoint['Smiles_Encoder_model_state_dict'])
            self.vocab= checkpoint['vocab']
        if output_layer:
            assert( output_layer >= 0 and output_layer <  self.Smiles_Encoder.encoder.num_layers), print(f'{output_layer} is invalid')
            self.output_layer= output_layer
        else:
            self.output_layer=self.Smiles_Encoder.encoder.num_layers-1
        self.pad_index= self.vocab.pad_index#checkpoint['pad_index']
        self.hidden_size= self.Smiles_Encoder.encoder.d_model
        self.seq_Generator=smiles_generator
        if self.seq_Generator:
            self.seq_Generator= checkpoint['seq_Generator']
            self.seq_Generator.load_state_dict(checkpoint['seq_Generator_model_state_dict'])
            self.seq_Generator.uncertainty  = torch.nn.Parameter(torch.zeros(1))
        if tasks:
            if type(tasks)==type({}):
                tasks=self.make_tasks(tasks)
            self.tasks   = nn.ModuleList(tasks)
            self.n_tasks= 1+len(tasks)
        if fine_tune_scale:
            assert fine_tune_scale >0 and fine_tune_scale<=1 , "fine_tune_scale should be in [0-1]"
            print(f'the learning rate of the base model is scaled by {fine_tune_scale}')
            parm_groups=[{ 'params': self.Smiles_Encoder.parameters(), 'lr': fine_tune_scale*lr }]
            if tasks:
                parm_groups+=[{ 'params': self.tasks.parameters(), 'lr': lr}]
            if self.seq_Generator:
                parm_groups+=[{ 'params': self.seq_Generator.parameters(), 'lr': lr }]
            self.optimizer = optim.Adam(parm_groups, lr=lr)
        else:
            print(f'freezing the parameters of the base model')
            for p in self.Smiles_Encoder.parameters(): p.requires_grad = False
            #if self.seq_Generator:for p in self.seq_Generator.parameters(): p.requires_grad = False
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
    ############################
    def forward(self, src,tgt=None):
        out={}
        device = src.device
        return_output_layers=[]
        en_out=self.Smiles_Encoder(src, return_output_layers=return_output_layers,
               return_atten_layers=[],
                       return_heads_layers=[])
        memory= en_out[f'{self.output_layer}']['output']
        hidden=memory[0, :].unsqueeze(0)
        if self.tasks:
            for task in self.tasks:
                out[task.task_name]=task(hidden)
        if  torch.is_tensor(tgt) and self.seq_Generator:
            tgt_embedded= self.Smiles_Encoder.Embeddings(tgt)
            tgt_key_padding_mask = torch.t(tgt).eq(self.pad_index).to(device)
            tgt_len= tgt.size()[0]
            tgt_mask = self.seq_Generator.decoder.generate_square_subsequent_mask(tgt_len).to(device)
            seq_out =self.seq_Generator(tgt_embedded, hidden,tgt_key_padding_mask=tgt_key_padding_mask, tgt_mask=tgt_mask) # (T,B,V)
            out[self.seq_Generator.task_name]=seq_out
        return out
    def make_tasks(self, tasks_dict):
        '''
        example:
        inputs: tasks_dict={
                    'chemaxon_LogD' :{'is_regression':1 , 'n_output':1,'n_layers': 1},
                    'pic50':{'is_regression':0 , 'n_output':1,'n_layers': 1}}
        output:
        tasks list
        '''
        hidden_size=self.hidden_size
        #if self.pool:        hidden_size=hidden_size
        tasks=[]
        for task_name in tasks_dict:
            try: pred_hidden_dropout=tasks_dict[task_name]['pred_hidden_dropout']
            except:pred_hidden_dropout=0.1
            try: out_dim=tasks_dict[task_name]['out_dim']
            except:out_dim=None
            if  tasks_dict[task_name]['is_regression']:
                model= regression_head(task_name=task_name,
                                        hidden_size=hidden_size,
                                      n_layers=tasks_dict[task_name]['n_layers'],
                                      n_output=tasks_dict[task_name]['n_output'],
                                      pred_hidden_dropout=pred_hidden_dropout, out_dim=out_dim)
                #tasks_dict[task_name]['loss_fn']= torch.nn.MSELoss(reduction='none')
            else:#Classification_head
                try:  ignore_index=tasks_dict[task_name]['mask']
                except:ignore_index= -100
                try: labels_id2name=tasks_dict[task_name]['labels_id2name']
                except: labels_id2name=None
                model= Classification_head(task_name=task_name,
                                        hidden_size=self.hidden_size,
                                      n_layers=tasks_dict[task_name]['n_layers'],
                                      n_output=tasks_dict[task_name]['n_output'],
                                      labels_id2name=labels_id2name,
                                      ignore_index=ignore_index,
                                      pred_hidden_dropout=pred_hidden_dropout)
            tasks.append(model)
        return tasks
    def truncate_tasks(self, tasks_to_keep):
        tasks =[]
        for task in self.tasks:
            if task.task_name in tasks_to_keep:
                tasks.append(task)
        self.tasks=nn.ModuleList(tasks)
    def predict(self, smiles,batch_size=50,prob=False,seq_len=220 ,class_label_id=False, convert_log10=True):
        device=next(self.parameters()).device
        def makebatch(sms):
            intgs=[]
            for s in sms:
                i= self.vocab.smile2int( s, max_smile_len=seq_len, with_eos=True, with_sos=True, return_len=False)
                intgs.append(torch.tensor(i))
            return torch.stack(intgs, dim=0)
        outputs={}
        for task in self.tasks:
            if task.__class__.__name__ == 'regression_head':
                outputs[f'predicted_{task.task_name}']=[]
            elif  task.__class__.__name__ == 'Classification_head':
                if class_label_id: outputs[f'predicted_{task.task_name}']=[]
                if prob: outputs[f'prob_{task.task_name}']=[]
                outputs[f'predicted_class_{task.task_name}']=[]
        st=0
        end=0
        while end < (len(smiles)):
            end= min(st+batch_size, len(smiles))
            src=makebatch(smiles[st:end])
            st+=batch_size
            src = src.to(device)
            src= torch.t(src)
            self.eval()
            with torch.no_grad():
                out= self.forward(src)
            for task in self.tasks:
                if task.__class__.__name__ == 'regression_head':
                    if task.n_output ==1:
                        outputs[f'predicted_{task.task_name}'].append(out[task.task_name].view(-1).detach().cpu().numpy() )
                    else:
                        outputs[f'predicted_{task.task_name}'].append(out[task.task_name].view(-1, task.n_output).detach().cpu().numpy() )
                elif task.__class__.__name__ == 'Classification_head':
                    probs_flatten=out[task.task_name].view(-1, task.n_output)
                    predicted_labels = torch.argmax(probs_flatten, dim=-1).view(-1).detach().cpu().numpy()
                    if class_label_id: outputs[f'predicted_{task.task_name}'].append(predicted_labels)
                    labels=  np.array([task.labels_id2name.get(i) for i in predicted_labels])
                    #outputs[f'predicted_{task.task_name}'].append(predicted_labels)
                    outputs[f'predicted_class_{task.task_name}'].append(labels)
                    if prob:
                        outputs[f'prob_{task.task_name}'].append(probs_flatten.detach().cpu().numpy())
        outputs = {key: np.concatenate(value, axis=0) for key, value in outputs.items()}
        if convert_log10:
            logk= [k for k in outputs.keys() if k.startswith('predicted_log10') ]
            if len(logk):
                for p in logk:
                    tp='_'.join(p.split('_')[2:])
                    outputs[f'predicted_{tp}']=10**outputs[p]
                    outputs.pop(p)
        return outputs
##############################################

################################################################3

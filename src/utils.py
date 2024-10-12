#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2022-12-21 15:30:08
LastEditTime: 2023-01-20 17:36:32
LastEditors: zharui@baidu.com
FilePath: /zharui/linkedin_embedding/src6/utils.py
Description: 
"""

import torch
import dgl
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence


def negtive_sampler(pos, ntype, num_ntype, neg_num=50):
    """ Get negative samples from current input to calculate the sequence-node contrast loss.

    Parameters
    ----------
    pos
        positive IDs, [N, ]
    ntype
        ID types, [N, ]
    num_ntype
        node types number
    neg_num, optional
        negative number, by default 50

    Returns
    -------
        the negative nodes for input sequences, [N, neg_num]
    """
    full_id, full_neg = [], []
    for item in range(num_ntype):
        index = torch.where(ntype == item)
        nodes = pos[index].numpy()
        neg_nodes = np.random.choice(nodes, size=(len(nodes), neg_num), replace=True)
        full_id.append(index[0])
        full_neg.append(torch.LongTensor(neg_nodes))
    
    full_id = torch.cat(full_id)
    full_neg = torch.cat(full_neg, dim=0)

    return full_neg[torch.sort(full_id)[1]]


def graph_to_sequence_block(g, seq, seq_len, block=None, frontier=None):
    """ Construct the bipartite-structured *block* for message passing from graph nodes to sequence entities.
    
        * The result block neglects the edges towards node 0, which we used to represent unknown experience.
        * All the input should be put on the CPU.
        * Can be accelerated !
    
    Parameters
    ----------
    g
        the entire graph.
    seq
        sequence batch
    seq_len
        list of sequence lengths of each batch element 
    block, optional
        block that contains the edges towards entities in sequence, by default None
    frontier, optional
        frontier structure data that contains the edges towards entities in sequence, by default None

    Returns
    -------
    res_block
        a new block for message passing from Graph to Sequence.
    valid_indices
        the index of nonzero enetities in the sequence.

    Raises
    ------
    ValueError
        no edges are given
    """
    if block:
        src, dst = g.find_edges(block.edata[dgl.EID])
        date = block.edata['date']
        weight = block.edata['weight']
        etype = block.edata[dgl.ETYPE]
    elif frontier:
        src, dst = frontier.edges()
        date = frontier.edata['date']
        weight = frontier.edata['weight']
        etype = frontier.edata[dgl.ETYPE]
    else:
        raise ValueError("Input Error.")

    zero_mask = dst != 0
    src = src[zero_mask]
    dst = dst[zero_mask]
    
    if not isinstance(seq, torch.Tensor):
        seq = torch.tensor(seq)
    
    # relabel the index
    seq_entity = pack_padded_sequence(seq.unsqueeze(-1), seq_len, batch_first=True).data.flatten()
    _, indices = torch.unique(torch.cat((dst, seq_entity)), return_inverse=True)
    rel_dst = indices[:len(dst)]
    rel_seq_entity = indices[len(dst):]
    
    # M_dst(i,j): i-th edge corresponds to j-ID
    M_dst_indices = torch.cat((torch.arange(len(rel_dst))[None, :], rel_dst[None, :]), dim=0)
    M_dst = torch.sparse_coo_tensor(indices=M_dst_indices, 
                                    values=torch.ones(len(dst)), 
                                    size=(len(dst), rel_seq_entity.max()+1))
    
    # M_seq(i,j): i-ID corresponds to j-th entity in sequence 
    seq_entity_index = torch.arange(seq.shape[0] * seq.shape[1]).reshape(seq.T.shape).T           
    valid_indices = pack_padded_sequence(seq_entity_index.unsqueeze(-1), seq_len, batch_first=True).data.squeeze()       
    _, rel_valid_indices = torch.unique(valid_indices, return_inverse=True)
    
    M_seq_indices = torch.cat((rel_seq_entity[None, :], rel_valid_indices[None, :]), dim=0)
    M_seq = torch.sparse_coo_tensor(indices=M_seq_indices, 
                                    values=torch.ones(len(seq_entity)), 
                                    size=(rel_seq_entity.max()+1, rel_valid_indices.max()+1))
    
    # M_{i,j} == 1 means that i-th edge corresponds to j-th entity in sequence
    M = torch.mm(M_dst, M_seq.to_dense())
    row, col = M.nonzero().T               
    col = valid_indices[col]
    
    # M1 = M.flatten() * torch.arange(1, M.shape[0] * M.shape[1] + 1)
    # M1 = (M1[M1 > 0] - 1).long()
    # row = M1 // M.shape[1]
    # col = M1 - row * M.shape[1]
    
    assert sum(dst[row] - seq.T.flatten()[col]) == 0 
    
    # src: absolute ID;  dst: i-th entity in sequence
    trans_g = dgl.graph((src[row], -col-1))
    trans_g.edata['date'] = date[zero_mask][row]
    trans_g.edata['weight'] = weight[zero_mask][row]
    trans_g.edata[dgl.ETYPE] = etype[zero_mask][row]

    res_block = dgl.to_block(trans_g, -valid_indices-1)
    res_block.dstdata['ntype'] = g.ndata['ntype'][seq_entity]
    
    return res_block, valid_indices


###### For Data Structure Transform Caused by DataLoader ######

def block2dict(block):
    """ Convert a block to dictionary data """
    return {"edges": block.edges(),
            "srcdata": block.srcdata,
            "dstdata": block.dstdata,
            "edata": block.edata,
            "num_src_nodes": block.num_src_nodes(),
            "num_dst_nodes": block.num_dst_nodes()}
    
    
def dict2block(block_dict):
    """ Transform dictionary data into block """
    edges = tuple([x.squeeze() for x in block_dict['edges']])
    block = dgl.create_block(data_dict=edges, 
                             num_src_nodes=int(block_dict['num_src_nodes']), 
                             num_dst_nodes=int(block_dict['num_dst_nodes']))
    
    for key, value in block_dict['srcdata'].items():
        block.srcdata[key] = value.squeeze()
    for key, value in block_dict['dstdata'].items():
        block.dstdata[key] = value.squeeze()
    for key, value in block_dict['edata'].items():
        block.edata[key] = value.squeeze()
    return block


def sbatch_convert(s_batch):
    """ Convert Sequence Batch """
    convert_ = dict()
    for key, value in s_batch.items():
        if isinstance(value, torch.Tensor):
            convert_[key] = value.squeeze()
        elif key == "block":
            convert_[key] = dict2block(value)
        else:
            raise NotImplemented
    return convert_


def gbatch_convert(g_batch):
    """ Convert Graph Batch """
    convert_ = dict()
    for key, value in g_batch.items():
        if isinstance(value, torch.Tensor):
            convert_[key] = value.squeeze()
        elif key == 'sub_g':
            convert_[key] = [dict2block(block_dict) for block_dict in value]
        elif key == 's_block':
            convert_[key] = dict2block(value)
        else:
            raise NotImplemented
    return convert_

    
if __name__ == "__main__":
    ntype = torch.randint(0, 2, (7,))
    x = torch.arange(7)
    print(x, ntype)

    print(negtive_sampler(x, ntype, 2, 3))
    

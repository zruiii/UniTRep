import math
import time
import torch
import numpy as np
from itertools import chain
from utils import graph_to_sequence_block, dict2block, block2dict

import dgl
from dgl.sampling import sample_neighbors
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, IterableDataset


class NegativeSampler:
    def __init__(self, k=10):  # negative sampling rate = 10
        self.k = k

    def sample(self, pos_samples, num_nodes):
        """ Negative Sampler for calculating Link Prediction Loss
        
        Parameters
        ----------
        pos_samples
            正边 (src, rel, dst, time)
        num_nodes
            节点数目

        Returns
        -------
            _description_
        """
        pos_samples = pos_samples[(pos_samples[:, 0] != 0) & (pos_samples[:, 2] != 0)]
        pos_samples = pos_samples[pos_samples[:, 0] != pos_samples[:, 2]]

        batch_size = len(pos_samples)
        neg_batch_size = batch_size * self.k
        neg_samples = np.tile(pos_samples, (self.k, 1))

        values = np.random.randint(pos_samples[:, [0, 2]].min(), num_nodes, size=neg_batch_size)
        choices = np.random.uniform(size=neg_batch_size)
        subj = choices > 0.5
        obj = choices <= 0.5
        neg_samples[subj, 0] = values[subj]
        neg_samples[obj, 2] = values[obj]
        samples = np.concatenate((pos_samples, neg_samples))

        labels = np.zeros(batch_size * (self.k + 1), dtype=np.float32)
        labels[:batch_size] = 1

        return torch.from_numpy(samples), torch.from_numpy(labels)


class MergeDataset(IterableDataset):
    """ 每次采样一个 batch 的序列，然后基于序列中包含的节点在全图中采样子图 """

    def __init__(self, 
                 parent_g, 
                 child_g, 
                 sequence, 
                 pool_map,
                 neg_num=10,
                 batch_size=500, 
                 drop_last=True,
                 graph_to_seq=True):
        
        self.parent_g = parent_g
        self.child_g = child_g
        self.sequence = sequence
        self.pool_map = pool_map
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.graph_to_seq = graph_to_seq
        
        self.n_samples = len(sequence)
        self.neg_sampler = NegativeSampler(k=neg_num)
        if not drop_last:
            self.n_batch = math.ceil(len(sequence) / batch_size)
        else:
            self.n_batch = int(len(sequence) / batch_size)
        self.indices = np.random.permutation(self.n_samples)
        self.child_ntype = self.child_g.ndata['ntype'].numpy()


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.n_batch
        else:
            per_worker = int(math.ceil(self.n_batch / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.n_batch)
        
        np.random.shuffle(self.indices[iter_start * self.batch_size: min(iter_end * self.batch_size, self.n_samples)])
        for it in range(iter_start, iter_end):
            yield self.get_batch(it)


    def __len__(self):
        return self.n_batch


    def get_batch(self, it):
        """ Get information for each batch. """
        _index = self.indices[self.batch_size * it: min(self.batch_size * (it + 1), self.n_samples)]
        
        ##### construct subsequence #####
        seq_batch = [self.sequence[idx] for idx in _index]
        seq_length = list(map(len, seq_batch))
        res_batch = np.zeros((len(seq_batch), max(seq_length)))    # Seq: [B, L]
        stime_batch = np.zeros((len(seq_batch), max(seq_length)))  # StartTime: [B, L]
        etime_batch = np.zeros((len(seq_batch), max(seq_length)))  # EndTime: [B, L]
        
        for idx, (seq, length) in enumerate(zip(seq_batch, seq_length)):
            res_batch[idx][:length] = [x[0] for x in seq]
            stime_batch[idx][:length] = [x[1] for x in seq]
            etime_batch[idx][:length] = [x[2] for x in seq]
        
        # sort by length
        sorted_id = sorted(range(len(seq_length)), key=lambda k: seq_length[k], reverse=True)
        seq_length = np.array(seq_length, dtype=int)[sorted_id]
        res_batch = np.array(res_batch[sorted_id], dtype=int)
        stime_batch = np.array(stime_batch[sorted_id], dtype=int)
        etime_batch = np.array(etime_batch[sorted_id], dtype=int)
        dur_batch = etime_batch - stime_batch
        ntype_batch = self.child_ntype[res_batch]

        s_batch = {
            'sub_s': res_batch,     # absolute ID
            'sub_ntype': ntype_batch,
            'seq_len': seq_length,
            'seq_dur': dur_batch,
            'seq_stime': stime_batch,
            'seq_etime': etime_batch
        }

        ##### construct subgraph #####
        g_batch = self.seq2graph(seq_batch)
        if self.graph_to_seq:
            block, valid_indices = graph_to_sequence_block(g=self.child_g, 
                                                        seq=res_batch, 
                                                        seq_len=seq_length, 
                                                        block=dict2block(g_batch['sub_g'][-1]))
            g_batch['s_block'] = block2dict(block)
            s_batch['valid_id'] = valid_indices
        
        return g_batch, s_batch


    def seq2graph(self, seq_batch):
        """ 基于序列中包含的节点, 从大图中采样其 2-hop 邻居(入边)构造子图
        
        Parameters
        ----------
        seq_batch
            一个 batch 序列所包含的节点

        Returns
        -------
            g_batch
        """
        node_set = set(chain(*[list(map(lambda x: x[0], item)) for item in seq_batch]))
        node_set = torch.LongTensor(sorted(node_set))

        # sample neighborhoods, the top-N nodes of 'seed_nodes' are 'node_set'
        blocks = []
        for idx, fanout in enumerate([5, 10]):
            if idx == 0:
                frontier = sample_neighbors(self.child_g, node_set, fanout=fanout)
                block = dgl.to_block(frontier, node_set)
            else:
                frontier = sample_neighbors(self.child_g, seed_nodes, fanout=fanout)
                block = dgl.to_block(frontier, seed_nodes)

            block.edata[dgl.EID] = frontier.edata[dgl.EID]
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        blocks = [block2dict(block) for block in blocks]
        
        # relabel & sample negative pairs for child edges: [src, r, dst, t]
        fn = lambda item: [(item[i][0], item[i + 1][0], item[i + 1][1]) for i in range(len(item) - 1)]
        child_edges = np.array(list(set(chain(*list(map(fn, seq_batch))))))         # drop duplicates
        child_src, child_dst = child_edges[:, 0], child_edges[:, 1]
        child_eid = self.child_g.edge_ids(child_src, child_dst)
        child_rel = self.child_g.edata[dgl.ETYPE][child_eid].numpy()
        child_nodes, relabel_child_edges = np.unique(child_edges[:, :2], return_inverse=True)
        relabel_child_edges = np.insert(relabel_child_edges.reshape(-1, 2), 1, child_rel, axis=1)
        relabel_child_edges = np.concatenate((relabel_child_edges, child_edges[:, 2].reshape(-1, 1)), axis=1)
        res_child_edges, res_child_labels = self.neg_sampler.sample(relabel_child_edges, child_nodes.shape[0])

        # parent edges: [src, r, dst, t]
        c2p = self.pool_map.t().coalesce().indices().T      # [E, 2] (child_ID, parent_ID)
        parent_src = c2p[child_src, :][:, 1].numpy()
        parent_dst = c2p[child_dst, :][:, 1].numpy()
        # parent_edges = np.array(list(set(zip(parent_src, parent_dst))))     # drop duplicates
        parent_edges = np.array(list(zip(parent_src, parent_dst)))
        parent_eid = self.parent_g.edge_ids(parent_edges[:, 0], parent_edges[:, 1])
        parent_rel = self.parent_g.edata[dgl.ETYPE][parent_eid].numpy()
        parent_nodes, relabel_parent_edges = np.unique(parent_edges, return_inverse=True)
        relabel_parent_edges = np.insert(relabel_parent_edges.reshape(-1, 2), 1, parent_rel, axis=1)
        relabel_parent_edges = np.concatenate((relabel_parent_edges, child_edges[:, 2].reshape(-1, 1)), axis=1)
        res_parent_edges, res_parent_labels = self.neg_sampler.sample(relabel_parent_edges, parent_nodes.shape[0])

        # mapping from relabel_child >> relabel_parent
        c2p_batch = c2p[node_set, :]  # [org_child_id, org_parent_id]
        map_src = np.unique(c2p_batch[:, 1], return_inverse=True)[1]
        map_dst = np.unique(c2p_batch[:, 0], return_inverse=True)[1]
        map_indices = torch.LongTensor(np.concatenate((map_src, map_dst)).reshape(2, -1))

        # todo: assert nodes
        
        g_batch = {
            'sub_g': blocks,
            'nids': seed_nodes,
            'c2p_map': map_indices,         # [rel_p, rel_c]
            'child_nodes': child_nodes,        # absolute
            'parent_nodes': parent_nodes,       # absolute
            'parent_edges': res_parent_edges,       # relabel
            'parent_labels': res_parent_labels,
            'child_edges': res_child_edges,         # relabel
            'child_labels': res_child_labels
        }
        return g_batch


class SeqDataset(IterableDataset):
    def __init__(self, g, sequence, batch_size=2000):
        self.g = g
        self.sequence = sequence
        self.batch_size = batch_size
        self.n_samples = len(sequence)
        self.n_batch = math.ceil(len(sequence) / batch_size)
        self.indices = np.random.permutation(self.n_samples)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.n_batch
        else:
            per_worker = int(math.ceil(self.n_batch / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.n_batch)
        
        np.random.shuffle(self.indices[iter_start * self.batch_size: min(iter_end * self.batch_size, self.n_samples)])
        for it in range(iter_start, iter_end):
            yield self.get_batch(it)

    def __len__(self):
        return self.n_batch

    def get_batch(self, it):
        """ 获取每个 batch 序列 """
        _index = self.indices[self.batch_size * it: min(self.batch_size * (it + 1), self.n_samples)]
        seq_batch = [self.sequence[idx] for idx in _index]
        seq_length = list(map(len, seq_batch))
        res_batch = np.zeros((len(seq_batch), max(seq_length)))         # nids
        dur_batch = np.zeros((len(seq_batch), max(seq_length)))         # duration
        stime_batch = np.zeros((len(seq_batch), max(seq_length)))       # start time
        etime_batch = np.zeros((len(seq_batch), max(seq_length)))       # end time
        for idx, (seq, length) in enumerate(zip(seq_batch, seq_length)):
            res_batch[idx][:length] = [x[0] for x in seq]
            stime_batch[idx][:length] = [x[1] for x in seq]
            etime_batch[idx][:length] = [x[2] for x in seq]

        # sort by length
        sorted_id = sorted(range(len(seq_length)), key=lambda k: seq_length[k], reverse=True)
        seq_length = np.array(seq_length, dtype=int)[sorted_id]
        res_batch = np.array(res_batch[sorted_id], dtype=int)
        stime_batch = np.array(stime_batch[sorted_id], dtype=int)
        etime_batch = np.array(etime_batch[sorted_id], dtype=int)
        dur_batch = etime_batch - stime_batch

        ############ Graph Neighbors of Sequence Entities ############
        node_set = torch.LongTensor(np.unique(res_batch))
        block, valid_indices = graph_to_sequence_block(g=self.g, 
                                                       seq=res_batch, 
                                                       seq_len=seq_length, 
                                                       frontier=sample_neighbors(self.g, node_set, fanout=-1))
        
        s_batch = {
            'sub_s': res_batch,
            'block': block2dict(block),
            'dur': dur_batch,
            'stime': stime_batch,
            'etime': etime_batch,
            'seq_len': seq_length,
            'valid_id': valid_indices
        }
        return s_batch


class ValSeqDataset(IterableDataset):
    def __init__(self, g, data, batch_size=2000, mode="test"):
        self.g = g
        self.sequence = data[f'{mode}_seq']
        self.dst = data[f'{mode}_dst']
        self.neg_dst = data[f'{mode}_neg_dst']
        self.batch_size = batch_size
        self.n_samples = len(self.sequence)
        self.n_batch = math.ceil(len(self.sequence) / batch_size)
        self.indices = np.random.permutation(self.n_samples)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.n_batch
        else:
            per_worker = int(math.ceil(self.n_batch / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.n_batch)
        
        np.random.shuffle(self.indices[iter_start * self.batch_size: min(iter_end * self.batch_size, self.n_samples)])
        for it in range(iter_start, iter_end):
            yield self.get_batch(it)

    def __len__(self):
        return self.n_batch

    def get_batch(self, it):
        """ 获取每个 batch 序列 """
        _index = self.indices[self.batch_size * it: min(self.batch_size * (it + 1), self.n_samples)]
        seq_batch = [self.sequence[idx] for idx in _index]
        seq_dst = [self.dst[idx] for idx in _index]
        seq_neg_dst = [self.neg_dst[idx] for idx in _index]
        
        seq_length = list(map(len, seq_batch))
        res_batch = np.zeros((len(seq_batch), max(seq_length)))         # nids
        dur_batch = np.zeros((len(seq_batch), max(seq_length)))         # duration
        stime_batch = np.zeros((len(seq_batch), max(seq_length)))       # start time
        etime_batch = np.zeros((len(seq_batch), max(seq_length)))       # end time
        
        for idx, (seq, length) in enumerate(zip(seq_batch, seq_length)):
            res_batch[idx][:length] = [x[0] for x in seq]
            stime_batch[idx][:length] = [x[1] for x in seq]
            etime_batch[idx][:length] = [x[2] for x in seq]

        # sort by length
        sorted_id = sorted(range(len(seq_length)), key=lambda k: seq_length[k], reverse=True)
        seq_dst = np.array(seq_dst, dtype=int)[sorted_id]
        seq_neg_dst = np.array(seq_neg_dst, dtype=int)[sorted_id]
        seq_length = np.array(seq_length, dtype=int)[sorted_id]
        res_batch = np.array(res_batch[sorted_id], dtype=int)
        stime_batch = np.array(stime_batch[sorted_id], dtype=int)
        etime_batch = np.array(etime_batch[sorted_id], dtype=int)
        dur_batch = etime_batch - stime_batch

        ############ Graph Neighbors of Sequence Entities ############
        node_set = torch.LongTensor(np.unique(res_batch))
        block, valid_indices = graph_to_sequence_block(g=self.g, 
                                                       seq=res_batch, 
                                                       seq_len=seq_length, 
                                                       frontier=sample_neighbors(self.g, node_set, fanout=-1))
        # todo: construct the block cost too much time
        
        s_batch = {
            'sub_s': res_batch,
            'block': block2dict(block),
            'dur': dur_batch,
            'stime': stime_batch,
            'etime': etime_batch,
            'seq_len': seq_length,
            'valid_id': valid_indices,
            'dst': seq_dst,
            'neg_dst': seq_neg_dst
        }
        return s_batch



    
    
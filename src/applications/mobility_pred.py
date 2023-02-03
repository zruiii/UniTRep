import models
from dataset import HierDataset
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from dgl.sampling import sample_neighbors

import os
import math
import random
import dgl
import tqdm
from torch.utils.data import IterableDataset
from utils import sbatch_convert, negtive_sampler, block2dict, graph_to_sequence_block

from collections import Counter
from sklearn.metrics import ndcg_score
from sklearn.metrics import label_ranking_average_precision_score


class SeqDataset(IterableDataset):
    def __init__(self, g, data, batch_size=2000, mode="test", MAX_L=4):
        # only keep test sequence with length > 2
        self.sequence, self.dst = [], []
        for i, seq in enumerate(data[f'{mode}_seq']):
            if len(seq) > MAX_L:
                self.sequence.append(seq)
                self.dst.append(data[f'{mode}_dst'][i])
        
        self.g = g   
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


@torch.no_grad()
def traj_inference(model, g, pre_embed, data, device, bs=4000, L=4):
    X, Y = [], []
    s_dataloader = torch.utils.data.DataLoader(SeqDataset(g, data, batch_size=bs, mode="test", MAX_L=L), 
                                               num_workers=4, prefetch_factor=2)
    for it, s_batch in enumerate(tqdm.tqdm(s_dataloader, desc="Individual Trajectory Prediction")):
        s_batch = sbatch_convert(s_batch)
        
        ############ Batch Data ############
        block = s_batch['block'].to(device)
        seq = torch.LongTensor(s_batch['sub_s']).to(device)   # absolute ID
        dur = torch.LongTensor(s_batch['dur']).to(device)
        stime = torch.LongTensor(s_batch['stime']).to(device)
        etime = torch.LongTensor(s_batch['etime']).to(device)
        seq_len = torch.LongTensor(s_batch['seq_len'])
        batch_dst = torch.LongTensor(s_batch['dst']).to(device)
        block = s_batch['block'].to(device)
        
        ############ Historical Information ############
        seq_embed = model.embedding(seq)
        stime_embed = model.child_dia(seq, seq_embed, stime.unsqueeze(-1))
        etime_embed = model.child_dia(seq, seq_embed, etime.unsqueeze(-1))
        seq_embed = torch.cat((stime_embed, etime_embed, model.time2vec(dur)), dim=-1)
        seq_embed = F.relu(model.experience(seq_embed))
        
        h0, c0 = model.init_hidden(seq.shape[0])
        packed_input = pack_padded_sequence(seq_embed, seq_len, batch_first=True)
        packed_output, (_, _) = model.lstm1(packed_input, (h0, c0))
        his_embed = packed_output.data
        seq_embed, _ = pad_packed_sequence(packed_output, batch_first=True)
        seq_nodes = pack_padded_sequence(seq.unsqueeze(-1), seq_len, batch_first=True).data.squeeze()
        
        ############ Graph Interaction ############
        src = block.srcdata[dgl.NID][block.num_dst_nodes():]
        x = F.relu(model.conv2(g=block,
                               src_h=pre_embed[src], 
                               src_tw=model.child_dia.w[src],
                               src_tb=model.child_dia.b[src],
                               edge_h=model.edge_embedding(block.edata['weight']),
                               dst_h=pre_embed[seq_nodes],
                               dst_his=his_embed,
                               dst_tw=model.child_dia.w[seq_nodes],
                               dst_tb=model.child_dia.b[seq_nodes]))
        
        # update sequence embedding
        seq_embed = seq_embed.transpose(0, 1).reshape(-1, x.shape[-1])
        seq_embed[s_batch['valid_id']] = x
        seq_embed = seq_embed.reshape(seq.shape[1], seq.shape[0], x.shape[-1]).transpose(0, 1)
        
        packed_input = pack_padded_sequence(seq_embed, seq_len, batch_first=True)
        packed_output, (_, _) = model.lstm2(packed_input, (h0, c0))
        seq_embed, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        ############ leave-one-out ############
        last_index = torch.LongTensor(seq_len-1)[:, None].to(device)
        seq_embed = seq_embed.gather(1, last_index.unsqueeze(-1).repeat(1, 1, 128)) 
        
        X.append(seq_embed.squeeze())
        Y.append(batch_dst)
    
    traj_embedding = torch.cat(X, dim=0)
    target = torch.cat(Y)
    
    return traj_embedding, target


class MLP(nn.Module):
    def __init__(self, in_dim, h_dim, max_class):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.linear_1 = nn.Linear(in_dim, h_dim)    
        self.linear_3 = nn.Linear(h_dim, h_dim)
        self.linear_2 = nn.Linear(h_dim, max_class)
    
    def forward(self, x):
        x = F.relu(self.linear_1(x))
        # x = self.dropout(x)
        # x = F.relu(self.linear_3(x))
        out = self.linear_2(x)
        return torch.softmax(out, dim=1)
        

def setup_seed(seed):
    """ Setup Random Seed """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def job2comp(job_id, job_map, comp_map):
    comp_id = []
    reverse_job_map = {value: key for key, value in job_map.items()}
    for job in tqdm.tqdm(job_id):
        job_name = reverse_job_map[int(job)]
        comp_name = job_name.split(',', 1)[0]
        comp_id.append(comp_map[comp_name])
    return comp_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ntypes", type=int, default=2)
    parser.add_argument("--num_rels", type=int, default=4)
    parser.add_argument("--bs", type=int, default=512)
    parser.add_argument("--cuda", type=int, default=5)
    parser.add_argument("--model", default='UniTRep')
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--t2v_dim", type=int, default=64)
    parser.add_argument("--edge_dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)               # 1e-3, 1e-4, 1e-5
    parser.add_argument("-m", type=str, default="UniTRep")        # model
    parser.add_argument("-v", type=int, default=11)        # model
    parser.add_argument("-k", type=int, default=200)            # top frequent companies
    parser.add_argument("-dv", type=int, default=5) 
    parser.add_argument("-l", type=int, default=4) 
    parser.add_argument("-ep", type=int, default=100) 
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.cuda))
    setup_seed(2023)
    
    data_path = "/data/data_v5/"
    data = HierDataset(data_path=data_path, create=False, version=args.dv)
    node_type = data.train_child_g.ndata['ntype']
    
    for ep in range(80, 101, 5):
        model = f"{args.m}_{args.v}/{args.m}_{ep}.pth"
        print(model)
        model_path = f"/src6/model_save/{model}"
        checkpoint = torch.load(model_path, map_location='cpu')
        GNN = getattr(models, args.model)
        model = GNN(parent_num=data.train_parent_g.num_nodes(), 
                    major_num=len(data.major_map), 
                    job_num=len(data.job_map),
                    h_dim=args.hidden_dim,
                    e_dim=args.edge_dim,
                    t2v_dim=args.t2v_dim,
                    num_ntypes=args.num_ntypes,
                    num_rels=args.num_rels,
                    num_heads=args.num_heads,
                    gamma=args.gamma,
                    device=device).to(device)

        model.load_state_dict(checkpoint['state_dict'])
        
        for l in [4]:
            all_embedding, target = traj_inference(model, data.train_child_g, checkpoint['pre_child_embed'].to(device), data.traj_pred, device, L=l)
            print("#############", l, all_embedding.shape, "#############")
            
            # only keep job targets
            target_type = node_type[target]
            all_embedding = all_embedding[target_type == 1]
            job_target = target[target_type == 1]
            comp_target = job2comp(job_target, data.job_map, data.company_map)
            
            # predict next company
            for k in [700]:
            # for k in [args.k]:
                print(f"####### K={k}; L={l} #######")
                
                top_comp = set([x[0] for x in Counter(comp_target).most_common(k)])
                target_index = [comp_target.index(x) for x in comp_target if x in top_comp]
                traj_embedding = all_embedding[target_index]
                target = torch.LongTensor(comp_target)[target_index]
                print(f"trajectories: {len(target_index)}; target company: {k}")
                
                # target re-index
                _, target = torch.unique(target, return_inverse=True)
                
                indices = np.random.permutation(traj_embedding.shape[0])
                train_index = int(len(indices) * 0.6)
                train_X = traj_embedding[:train_index].to(device)
                train_y = target[:train_index].to(device)

                test_X = traj_embedding[train_index:].to(device)
                test_y = target[train_index:]

                n_batch = math.ceil(train_index / args.bs)
                predicor = MLP(128, 128, int(target.max())+1).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(predicor.parameters(), lr=args.lr, weight_decay=1e-4)
                
                ndcg_10, ndcg_50, recall_10, recall_50 = 0, 0, 0, 0
                for ep in range(100):
                    indices = np.random.permutation(train_index)
                    
                    ############ Training ############
                    # for it in tqdm.tqdm(range(n_batch), desc=f"K = {k} -- Epoch {ep+1} Training"):
                    for it in range(n_batch):
                        _index = indices[it * args.bs : min((it + 1) * args.bs, train_index)]

                        x = train_X[_index]
                        y = train_y[_index]
                        
                        pred = predicor(x)
                        loss = criterion(pred, y)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    ############ Evaluate ############
                    with torch.no_grad():
                        score = predicor(test_X).cpu()
                        labels = torch.zeros((score.shape[0], k))
                        labels = labels.scatter(1, test_y[:, None], 1)
                        
                        ndcg_5 = ndcg_score(labels.cpu().numpy(), score.cpu().numpy(), k=5)
                        ndcg_20 = ndcg_score(labels.cpu().numpy(), score.cpu().numpy(), k=20)
                        ndcg_10 = ndcg_score(labels.cpu().numpy(), score.cpu().numpy(), k=10)
                        ndcg_50 = ndcg_score(labels.cpu().numpy(), score.cpu().numpy(), k=50)
                        
                        sort_score = torch.sort(score, dim=1, descending=True).values.cpu().numpy()
                        indices = torch.sort(score, dim=1, descending=True).indices
                        sort_labels = labels.gather(1, indices)

                        recall_5 = sort_labels[:, :5].sum() / sort_labels.shape[0]
                        recall_20 = sort_labels[:, :20].sum() / sort_labels.shape[0]
                        recall_10 = sort_labels[:, :10].sum() / sort_labels.shape[0]
                        recall_50 = sort_labels[:, :50].sum() / sort_labels.shape[0]
                        
                        mrr = label_ranking_average_precision_score(labels.cpu().numpy(), score.cpu().numpy())
                        
                        print(f"MRR: {mrr}")
                        print("Epoch: {}  NDCG@5: {:.4f}  NDCG@20: {:.4f}  Recall@5: {:.4f}  Recall@20: {:.4f}".format(ep+1, ndcg_5, ndcg_20, recall_5, recall_20))          
                        print("Epoch: {}  NDCG@10: {:.4f}  NDCG@50: {:.4f}  Recall@10: {:.4f}  Recall@50: {:.4f}".format(ep+1, ndcg_10, ndcg_50, recall_10, recall_50))          
                        print('\n')






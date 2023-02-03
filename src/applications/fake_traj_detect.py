import models
from dataset import HierDataset
import argparse
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import os

import dgl
import tqdm
from utils import sbatch_convert

import math
import random
from dgl.sampling import sample_neighbors
from torch.utils.data import DataLoader, IterableDataset
from utils import graph_to_sequence_block, dict2block, block2dict
from sklearn import linear_model
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score


def fake_generator(traj_set, job_map, major_map):
    job_set = list(job_map.values())
    major_set = list(major_map.values())
    for idx, traj in enumerate(traj_set):
        replace_id = np.random.choice(len(traj))
        node = traj[replace_id][0]
        while node == 0:
            replace_id = np.random.choice(len(traj))
            node = traj[replace_id][0]

        if node in job_set:
            replace_node = np.random.choice(job_set)
            while replace_node == node:
                replace_node = np.random.choice(job_set)
        elif node in major_set:
            replace_node = np.random.choice(major_set)
            while replace_node == node:
                replace_node = np.random.choice(major_set)
        else:
            raise ValueError

        traj_set[idx][replace_id] = (replace_node, traj[replace_id][1], traj[replace_id][2])     
    
    return traj_set
    

class SeqDataset(IterableDataset):
    def __init__(self, g, train_seq, batch_size=2000):
        self.g = g
        
        # random select k trajectories
        self.sequence = train_seq
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
        # todo: construct the block cost too much time
        
        s_batch = {
            'sub_s': res_batch,
            'block': block2dict(block),
            'dur': dur_batch,
            'stime': stime_batch,
            'etime': etime_batch,
            'seq_len': seq_length,
            'valid_id': valid_indices,
        }
        return s_batch



@torch.no_grad()
def traj_inference(model, g, pre_embed, data, device, bs=2000):
    X, Y = [], []
    s_dataloader = torch.utils.data.DataLoader(SeqDataset(g, data, batch_size=bs), num_workers=4, prefetch_factor=2)
    for it, s_batch in enumerate(tqdm.tqdm(s_dataloader, desc="Individual Trajectory Prediction")):
        s_batch = sbatch_convert(s_batch)
        
        ############ Batch Data ############
        block = s_batch['block'].to(device)
        seq = torch.LongTensor(s_batch['sub_s']).to(device)   # absolute ID
        dur = torch.LongTensor(s_batch['dur']).to(device)
        stime = torch.LongTensor(s_batch['stime']).to(device)
        etime = torch.LongTensor(s_batch['etime']).to(device)
        seq_len = torch.LongTensor(s_batch['seq_len'])
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
    
    traj_embedding = torch.cat(X, dim=0)
    
    # torch.save(torch.cat((traj_embedding, target[:, None]), dim=1), save_path)
    return traj_embedding
        



def detection(pos_traj_embedding, neg_traj_embedding):
    pos_label = np.zeros(pos_traj_embedding.shape[0])
    neg_label = np.ones(neg_traj_embedding.shape[0])
    pre_li, re_li, f1_li = [], [], []
    
    kf = KFold(n_splits=5, shuffle=True)
    pos_list = list(kf.split(pos_traj_embedding, pos_label))
    neg_list = list(kf.split(neg_traj_embedding, neg_label))

    for it in range(5):
        prun_pos_id = random.sample(list(pos_list[it][0]), int(1.25 * len(neg_list[it][0])))
        pos_embed = pos_traj_embedding[prun_pos_id]
        neg_embed = neg_traj_embedding[neg_list[it][0]]
        train_X = np.concatenate((pos_embed, neg_embed), axis=0)
        train_Y = np.concatenate((pos_label[prun_pos_id], neg_label[neg_list[it][0]]))

        classifier = linear_model.LogisticRegression(max_iter=500, n_jobs=10)
        classifier.fit(train_X, train_Y)
        
        test_X = np.concatenate((pos_traj_embedding[pos_list[it][1]], neg_traj_embedding[neg_list[it][1]]), axis=0)
        test_Y = np.concatenate((pos_label[pos_list[it][1]], neg_label[neg_list[it][1]]))

        pred = classifier.predict(test_X)
        precision = precision_score(test_Y, pred)
        recall = recall_score(test_Y, pred)
        f1 = f1_score(test_Y, pred, average='micro')
        
        print(precision, recall, f1)
        pre_li.append(precision)
        re_li.append(recall)
        f1_li.append(f1)
    
    return np.mean(pre_li), np.mean(re_li), np.mean(f1_li)


# note: 使用孤立森林
def detection2(pos_traj_embedding, neg_traj_embedding):
    train_pos_X = pos_traj_embedding[:int(pos_traj_embedding.shape[0] * 0.8)]
    train_neg_X = neg_traj_embedding[:int(neg_traj_embedding.shape[0] * 0.8)]
    best_p, best_r, best_f1 = 0, 0, 0
    
    model = IsolationForest(n_estimators=400, 
                            max_features=128,
                            n_jobs=10,
                            contamination=0.1,
                            random_state=2023)
    model.fit(np.concatenate((train_pos_X, train_neg_X), axis=0))

    test_neg_X = neg_traj_embedding[int(neg_traj_embedding.shape[0] * 0.8): ]
    test_pos_X = pos_traj_embedding[int(pos_traj_embedding.shape[0] * 0.8): int(pos_traj_embedding.shape[0] * 0.8) + test_neg_X.shape[0]]
    test_Y = np.concatenate((np.zeros(test_pos_X.shape[0]), np.ones(test_neg_X.shape[0])))
    
    pred = model.predict(np.concatenate((test_pos_X, test_neg_X), axis=0))
    pred[pred == 1] = 0
    pred[pred == -1] = 1
    precision = precision_score(test_Y, pred)
    recall = recall_score(test_Y, pred)
    # f1 = f1_score(test_Y, pred, average='micro')
    f1 = f1_score(test_Y, pred)
    
    print(precision, recall, f1)
    if precision + recall + f1 > best_p + best_r + best_f1:
        best_p = precision
        best_r = recall
        best_f1 = f1
    
    return best_p, best_r, best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ntypes", type=int, default=2)
    parser.add_argument("--num_rels", type=int, default=4)
    parser.add_argument("--bs", type=int, default=2000)
    parser.add_argument("--cuda", type=int, default=5)
    parser.add_argument("--model", default='UniTRep')
    parser.add_argument("--sub_v", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--t2v_dim", type=int, default=64)
    parser.add_argument("--edge_dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--neg_num", type=int, default=10)
    parser.add_argument("--contra_num", type=int, default=200)
    parser.add_argument("--mj_pool", type=str, default="attn")
    parser.add_argument("--traj_pool", type=str, default="attn")
    parser.add_argument("--seed", type=int, default=2023)               # 36, 42, 17, 24, 2023, 50, 66, 74, 81, 98
    parser.add_argument("--lr", type=float, default=1e-4)               # 1e-3, 1e-4, 1e-5
    parser.add_argument("--decay", type=float, default=1e-4)            # 1e-4, 1e-3, 5e-4
    parser.add_argument("--lam_tm", type=float, default=0.4)            # 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 (参敏实验)
    parser.add_argument("--propor_mj", type=int, default=3)             # 1, 3, 5, 7, 9
    parser.add_argument("--seq_loss", type=str, default="contra")
    parser.add_argument("--save", action='store_true')      # save model
    parser.add_argument("--log", action='store_true')   # tensorboard
    parser.add_argument("-m", type=str, default="UniTRep")        # model
    parser.add_argument("-v", type=int, default=11)        # model
    parser.add_argument("-k", type=int, default=20000)        # positive trajectories
    parser.add_argument("-w", action='store_true')
    parser.add_argument("-ep", type=int, default=100)        # model
    parser.add_argument("-dv", type=int, default=5) 
    args = parser.parse_args()
    device = torch.device('cpu'.format(args.cuda))
    
    
    data_path = "/data/data_v5/"
    data = HierDataset(data_path=data_path, create=False, version=args.dv)
    
    res_p, res_r, res_f = [], [], []
    for ep in range(50, 101, 5):
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
        
        # positive
        pos_seq = [x for x in data.train_seq if len(x) >= 4]
        pos_seq = random.sample(pos_seq, args.k) 
        pos_traj_embedding = traj_inference(model, data.train_child_g, checkpoint['pre_child_embed'].to(device), pos_seq, device)
        
        # negative
        neg_seq = fake_generator(random.sample(pos_seq, int(0.1 * args.k)), data.job_map, data.major_map)
        neg_traj_embedding = traj_inference(model, data.train_child_g, checkpoint['pre_child_embed'].to(device), neg_seq, device)
        
        pos_traj_embedding = pos_traj_embedding.detach().cpu().numpy()
        neg_traj_embedding = neg_traj_embedding.detach().cpu().numpy()

        precision, recall, f1 = detection2(pos_traj_embedding, neg_traj_embedding)
        print(f"Precision: {precision} | Recall: {recall} | F1: {f1}")
        
        res_p.append(precision)
        res_r.append(recall)
        res_f.append(f1)
    
    print(np.mean(res_p), np.std(res_p))
    print(np.mean(res_r), np.std(res_r))
    print(np.mean(res_f), np.std(res_f))



    



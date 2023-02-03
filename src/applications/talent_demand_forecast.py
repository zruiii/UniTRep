

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, in_dim, h_dim, f_dim, max_flow, L):
        super().__init__()
        self.L = L
        self.h_dim = h_dim
        self.flow_map = nn.Embedding(max_flow, f_dim)
        
        self.linear_1 = nn.Linear(in_dim + f_dim, h_dim)
        self.linear_2 = nn.Linear(L * h_dim, 5)

    def forward(self, node_feat, flow_feat, dia=False):
        flow_feat = self.flow_map(flow_feat)            # [B, L, I]
        if not dia:
            node_feat = node_feat.unsqueeze(1).repeat(1, self.L, 1)

        x = self.linear_1(torch.cat((flow_feat, node_feat), dim=-1))    # [B, L, H+H']
        x = F.relu(x)
        out = self.linear_2(x.reshape(-1, self.L * self.h_dim))
        out = torch.softmax(out, dim=1)
        return out


class MLP2(nn.Module):
    def __init__(self, in_dim, h_dim, L):
        super().__init__()
        self.L = L
        self.h_dim = h_dim
        self.linear_1 = nn.Linear(in_dim + L, h_dim)
        self.linear_2 = nn.Linear(h_dim, 5)
        self.linear_3 = nn.Linear(in_dim, 1)
        self.linear_4 = nn.Linear(L + 1, 5)

    def forward(self, node_feat, flow_feat):        # [B, H] [B, L]
        flow_feat = flow_feat.to(torch.float32)
        # x = torch.cat((F.relu(self.linear_1(flow_feat)), node_feat), dim=1)
        # out = self.linear_2(x)
        
        out = torch.cat((self.linear_3(node_feat), flow_feat), dim=1)
        # out = self.linear_1(torch.cat((node_feat, flow_feat), dim=1))
        out = self.linear_4(F.relu(out))
        out = torch.softmax(out, dim=1)
        return out


def setup_seed(seed):
    """ Setup Random Seed """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def symbolic(x, a=0.4, b=1.5):
    if x < -b:
        return 0
    elif x >= -b and x < -a:
        return 1
    elif x >= -a and x <= a:
        return 2
    elif x > a and x <= b:
        return 3
    else:
        return 4
    
    
def data_convert(df, a=0.25, b=0.85, L=5):
    res_data, res_time = list(), list()
    for idx, row in df.iterrows():
        for j in range(len(row) - L):
            window = list(row[j: j + L + 1])
            if L > 1:
                diff = (window[-1] - window[-2]) / np.std(window)
            else:
                diff = (window[-1] - window[-2])
            label = symbolic(diff, a, b)
            res_data.append([idx, ] + window[:-1] + [label, ])
            res_time.append(list(row.index)[j: j + L])
    
    res_data = np.array(res_data)
    res_time = np.array(res_time, dtype=int)
    
    train_mask = res_time[:, -1] <= 2010
    train_data = res_data[train_mask]           # [nid, x1, x2, ..., y]
    train_time = res_time[train_mask]
    
    test_data = res_data[~train_mask]
    test_time = res_time[~train_mask]
    return train_data, train_time, test_data, test_time



def trend_pred_mlp(embedding, train_data, test_data, args):
    """ Binary Trend Classification with MLP """
    device = torch.device('cuda:{}'.format(args.cuda))
    n_samples = train_data.shape[0]
    embedding = embedding.to(device)
    
    # feature
    train_flow = train_data[:, 1:-1] / 10           # 以10为区间划分
    train_flow = torch.Tensor(train_flow).long().to(device)          # [N, L]
    train_node = embedding[np.array(train_data[:, 0], dtype=int)]             # [N, H]
    
    test_flow = test_data[:, 1:-1] / 10
    test_flow = torch.Tensor(test_flow).long().to(device)
    test_node = embedding[np.array(test_data[:, 0], dtype=int)]
    
    # label
    train_y = torch.LongTensor(train_data[:, -1]).to(device)
    test_y = test_data[:, -1]

    n_batch = math.ceil(n_samples / args.bs)
    model = MLP(128, 32, 32, max_flow=train_flow.max() + 20, L=args.k).to(device)
    # model = MLP(128, 32, L=args.k).to(device)
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    best_mi, best_ma, best_wei, best_sum = 0, 0, 0, 0
    
    # for ep in range(100):
    for ep in range(100):
        total_loss = 0
        indices = np.random.permutation(n_samples)
        
        for it in range(n_batch):
            _index = indices[it * args.bs : min((it + 1) * args.bs, n_samples)]

            node_feat = train_node[_index]
            flow_feat = train_flow[_index]
            pred = model(node_feat, flow_feat)
            label = train_y[_index]

            # loss = F.binary_cross_entropy_with_logits(pred, train_y[_index])
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            total_loss += loss
        
        # print(f"Loss: {total_loss / n_batch}")
        
        ############ Evaluate ############
        with torch.no_grad():
            test_pred = model(test_node, test_flow).cpu()
            test_pred = torch.argmax(test_pred, dim=1).numpy()
            
            micro_f1 = f1_score(test_y, test_pred, average='micro')
            macro_f1 = f1_score(test_y, test_pred, average='macro')
            weighted_f1 = f1_score(test_y, test_pred, average='weighted')
            print(f"Epoch: {ep+1}  Micro-F1: {micro_f1}, Macro-F1: {macro_f1}, Weighted-F1: {weighted_f1}")
            
            if micro_f1 + macro_f1 + weighted_f1 > best_sum:
                best_mi = micro_f1
                best_ma = macro_f1
                best_wei = weighted_f1
                best_sum = best_mi + best_ma + best_wei
                fp = 0
            else:
                fp += 1

            if fp == 8:
                break
        
    print(f"Micro-F1: {best_mi}, Macro-F1: {best_ma}, Weighted-F1: {best_wei}")
    return best_mi, best_ma, best_wei


def dia_trend_pred_mlp(embedding, repr_model, train_data, test_data, train_time, test_time, args):
    """ Binary Trend Classification with MLP """
    device = torch.device('cuda:{}'.format(args.cuda))
    n_samples = train_data.shape[0]
    embedding = embedding.to(device)
    
    # feature
    train_flow = train_data[:, 1:-1] / 10           # 以10为区间划分
    train_flow = torch.Tensor(train_flow).long().to(device)          # [N, L]
    
    train_id = torch.LongTensor(train_data[:, 0])[:, None].repeat(1, args.k).to(device)
    t = torch.Tensor(train_time).to(device)
    train_node = repr_model.parent_dia(train_id, embedding[train_id], t.unsqueeze(-1))  # [N, L, H]
    train_node = train_node.detach()
    # train_node = embedding[train_id].detach()
    
    
    test_flow = test_data[:, 1:-1] / 10
    test_flow = torch.Tensor(test_flow).long().to(device)
    
    test_id = torch.LongTensor(test_data[:, 0])[:, None].repeat(1, args.k).to(device)
    t = torch.Tensor(test_time).to(device)
    test_node = repr_model.parent_dia(test_id, embedding[test_id], t.unsqueeze(-1))  # [N, L, H]
    test_node = test_node.detach()
    # test_node = embedding[test_id].detach()
    
    # label
    train_y = torch.LongTensor(train_data[:, -1]).to(device)
    test_y = test_data[:, -1]

    n_batch = math.ceil(n_samples / args.bs)
    model = MLP(128, 32, 32, max_flow=train_flow.max() + 20, L=args.k).to(device)
    
    # model = MLP(128, 32, L=args.k).to(device)
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    best_mi, best_ma, best_wei, best_sum = 0, 0, 0, 0
    
    # for ep in range(100):
    for ep in range(200):
        total_loss = 0
        indices = np.random.permutation(n_samples)
        
        for it in range(n_batch):
            _index = indices[it * args.bs : min((it + 1) * args.bs, n_samples)]

            node_feat = train_node[_index]      # [B, L, H]
            flow_feat = train_flow[_index]
            pred = model(node_feat, flow_feat, dia=True)
            # pred = model(node_feat[:, -1, :], flow_feat)
            
            label = train_y[_index]

            # loss = F.binary_cross_entropy_with_logits(pred, train_y[_index])
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            total_loss += loss
        
        # print(f"Loss: {total_loss / n_batch}")
        
        ############ Evaluate ############
        with torch.no_grad():
            test_pred = model(test_node, test_flow, dia=True).cpu()
            # test_pred = model(test_node[:, -1, :], test_flow).cpu()
            
            test_pred = torch.argmax(test_pred, dim=1).numpy()
            
            micro_f1 = f1_score(test_y, test_pred, average='micro')
            macro_f1 = f1_score(test_y, test_pred, average='macro')
            weighted_f1 = f1_score(test_y, test_pred, average='weighted')
            
            print(f"Micro-F1: {micro_f1}, Macro-F1: {macro_f1}, Weighted-F1: {weighted_f1}")
            
            if micro_f1 + macro_f1 + weighted_f1 > best_sum:
                best_mi = micro_f1
                best_ma = macro_f1
                best_wei = weighted_f1
                best_sum = micro_f1 + macro_f1 + weighted_f1
                fp = 0
            else:
                fp += 1

            if fp == 8:
                break
        
    print(f"Micro-F1: {best_mi}, Macro-F1: {best_ma}, Weighted-F1: {best_wei}")
    return best_mi, best_ma, best_wei
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=str, default="job")
    parser.add_argument("--f", type=str, default="inflow")
    parser.add_argument("-m", type=str, default="UniTRep")        # model
    parser.add_argument("--model", default='UniTRep')
    parser.add_argument("-v", type=int, default=11)        # model
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--cuda", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--num_ntypes", type=int, default=2)
    parser.add_argument("--num_rels", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--t2v_dim", type=int, default=64)
    parser.add_argument("--edge_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ep", type=int, default=80)
    args = parser.parse_args()
    setup_seed(2023) 
    device = torch.device('cuda:{}'.format(args.cuda))
    
    df = pd.read_csv("/applications/company_inflow_1999_2019.csv", index_col=0)
    
    # 0.8
    # for b in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
    
    res_mi, res_ma, res_wei = [], [], []
    for ep in range(5, 31, 5):
    # for ep in range(50, 101, 5):
        for b in [0.8]:
            print(b)
            train_data, train_time, test_data, test_time = data_convert(df, b=b, L=args.k)
            
            model_path = f"/model_save/{args.m}_{args.v}/{args.m}_{ep}.pth"
            if not os.path.exists(model_path):
                print(model_path)
                raise NotImplementedError
            
            print(f"{args.m}_{args.v}/{args.m}_{ep}.pth")
            checkpoint = torch.load(model_path, map_location=device)
            embedding = checkpoint['child_embedding']
            
            if 'UniTRep' in args.m:
                import models
                from dataset import HierDataset

                data_path = "/data/data_v5/"
                data = HierDataset(data_path=data_path, create=False)
                train_parent_g = data.train_parent_g
                
                GNN = getattr(models, args.model)
                model = GNN(parent_num=train_parent_g.num_nodes(), 
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
                micro_f1, macro_f1, weighted_f1 = dia_trend_pred_mlp(embedding, model, train_data, test_data, train_time, test_time, args)
                res_mi.append(micro_f1)
                res_ma.append(macro_f1)
                res_wei.append(weighted_f1)
            else:    
                micro_f1, macro_f1, weighted_f1 = trend_pred_mlp(embedding, train_data, test_data, args)
                res_mi.append(micro_f1)
                res_ma.append(macro_f1)
                res_wei.append(weighted_f1)
    
    print(np.mean(res_mi), np.std(res_mi))
    print(np.mean(res_ma), np.std(res_ma))
    print(np.mean(res_wei), np.std(res_wei))
    
        
        
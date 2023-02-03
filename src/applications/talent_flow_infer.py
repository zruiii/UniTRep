import numpy as np
import os
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import torch

from tqdm import tqdm
import argparse

import models
from dataset import HierDataset

def super_static_link_pred(embed, data, num_rels=4, mode="test", k=5, max_iter=1000):
    """ K-Fold cross-validation for Supervised Link Prediction Task """
    
    embed = embed.cpu().numpy()
    auc_list, ap_list, acc_list, counts = [], [], [], []
    
    kf = KFold(n_splits=k, shuffle=False)
    for rel in range(num_rels):
        pos_edges = data[rel][f'{mode}_pos_edges']      # [E, 3]
        neg_edges = data[rel][f'{mode}_neg_edges']
        counts.append(pos_edges.shape[0])
        
        print(f"Rel-Type: {rel} Evaluation")
        # k-flod for each edge type
        rel_auc, rel_ap, rel_acc = [], [], []
        for idx, (train_index, test_index) in enumerate(kf.split(pos_edges)):
            train_edges = np.concatenate((pos_edges[train_index], neg_edges[train_index]))
            train_labels = np.concatenate((np.ones(pos_edges[train_index].shape[0]),
                                           np.zeros(neg_edges[train_index].shape[0])))
            
            src_embed = embed[train_edges[:, 0]]
            dst_embed = embed[train_edges[:, 1]]
            X = np.concatenate((src_embed, dst_embed), axis=1)
            
            print(f"Rel: {rel}  Fold: {idx}")
            model = linear_model.LogisticRegression(max_iter=max_iter, n_jobs=10)
            model.fit(X, train_labels)
            
            test_edges = np.concatenate((pos_edges[test_index], neg_edges[test_index]))
            test_labels = np.concatenate((np.ones(pos_edges[test_index].shape[0]), 
                                          np.zeros(neg_edges[test_index].shape[0])))
            pred = model.predict(np.concatenate((embed[test_edges[:, 0]], embed[test_edges[:, 1]]), axis=1))
            prob = model.predict_proba(np.concatenate((embed[test_edges[:, 0]], embed[test_edges[:, 1]]), axis=1))[:, 1]

            acc = accuracy_score(test_labels, pred)
            auc = roc_auc_score(test_labels, prob)
            ap = average_precision_score(test_labels, prob)
            rel_auc.append(auc)
            rel_ap.append(ap)
            rel_acc.append(acc)
            print(f"AUC: {auc}  AP: {ap}  ACC: {ap}")
        
        auc_list.append(np.mean(rel_auc))
        ap_list.append(np.mean(rel_ap))
        acc_list.append(np.mean(rel_acc))
    
    acc_avg = sum([x[0] * x[1] for x in list(zip(counts, acc_list))]) / sum(counts)
    auc_avg = sum([x[0] * x[1] for x in list(zip(counts, auc_list))]) / sum(counts)
    ap_avg = sum([x[0] * x[1] for x in list(zip(counts, ap_list))]) / sum(counts)
    
    auc_list.append(auc_avg)
    ap_list.append(ap_avg)
    acc_list.append(acc_avg)

    return auc_list, ap_list, acc_list

@torch.no_grad()
def super_diachornic_link_pred(embed, data, model, num_rels=4, mode="test", k=5, max_iter=1000, m=None):
    auc_list, ap_list, acc_list, counts = [], [], [], []
    
    kf = KFold(n_splits=5, shuffle=False)
    for rel in range(num_rels):
        pos_edges = data[rel][f'{mode}_pos_edges'][:m]      # [E, 3]
        neg_edges = data[rel][f'{mode}_neg_edges'][:m]
        counts.append(pos_edges.shape[0])
        
        # k-flod for each edge type
        print(f"Rel-Type: {rel} Evaluation")
        rel_auc, rel_ap, rel_acc = [], [], []
        for idx, (train_index, test_index) in enumerate(kf.split(pos_edges)):
            train_edges = np.concatenate((pos_edges[train_index], neg_edges[train_index]))
            train_labels = np.concatenate((np.ones(pos_edges[train_index].shape[0]),
                                           np.zeros(neg_edges[train_index].shape[0])))
            
            print(f"Rel: {rel}  Fold: {idx}")
            src = torch.LongTensor(train_edges[:, 0])
            dst = torch.LongTensor(train_edges[:, 1])
            t = torch.Tensor(train_edges[:, 2])[:, None]
            
            src_embed = model.child_dia(src, embed[src], t) 
            dst_embed = model.child_dia(dst, embed[dst], t)
            X = torch.cat((src_embed, dst_embed), dim=1).numpy() 
            
            classifier = linear_model.LogisticRegression(max_iter=max_iter, n_jobs=10)
            classifier.fit(X, train_labels)
            
            test_edges = np.concatenate((pos_edges[test_index], neg_edges[test_index]))
            test_labels = np.concatenate((np.ones(pos_edges[test_index].shape[0]), 
                                          np.zeros(neg_edges[test_index].shape[0])))
            
            src = torch.LongTensor(test_edges[:, 0])
            dst = torch.LongTensor(test_edges[:, 1])
            t = torch.Tensor(test_edges[:, 2])[:, None]
            
            src_embed = model.child_dia(src, embed[src], t) 
            dst_embed = model.child_dia(dst, embed[dst], t)
            X = torch.cat((src_embed, dst_embed), dim=1).numpy() 
            pred = classifier.predict(X)
            prob = classifier.predict_proba(X)[:, 1]
            
            acc = accuracy_score(test_labels, pred)
            auc = roc_auc_score(test_labels, prob)
            ap = average_precision_score(test_labels, prob)
            rel_auc.append(auc)
            rel_ap.append(ap)
            rel_acc.append(acc)
            print(f"AUC: {auc}  AP: {ap}  ACC: {ap}")
            
            # 1-fold
            if idx == 0:
                break
        
        auc_list.append(np.mean(rel_auc))
        ap_list.append(np.mean(rel_ap))
        acc_list.append(np.mean(rel_acc))
        
    acc_avg = sum([x[0] * x[1] for x in list(zip(counts, acc_list))]) / sum(counts)
    auc_avg = sum([x[0] * x[1] for x in list(zip(counts, auc_list))]) / sum(counts)
    ap_avg = sum([x[0] * x[1] for x in list(zip(counts, ap_list))]) / sum(counts)
    
    auc_list.append(auc_avg)
    ap_list.append(ap_avg)
    acc_list.append(acc_avg)

    return auc_list, ap_list, acc_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ntypes", type=int, default=2)
    parser.add_argument("--num_rels", type=int, default=4)
    parser.add_argument("--bs", type=int, default=2000)
    parser.add_argument("--cuda", type=int, default=7)
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
    parser.add_argument("-m", type=str, default="UniTRep_TE")        # model
    parser.add_argument("-v", type=int, default=1)        # model
    parser.add_argument("-i", type=int, default=500) 
    parser.add_argument("-ep", type=int, default=80) 
    parser.add_argument("-dv", type=int, default=5) 
    args = parser.parse_args()
    device = torch.device('cpu'.format(args.cuda))
    
    res_auc, res_ap, res_acc = [], [], []
    # for ep in range(50, 101, 5):
    for ep in range(20, 51, 5):
        model = f"{args.m}_{args.v}/{args.m}_{ep}.pth"
        data_path = "/data/data_v5/"
        model_path = f"/src6/model_save/{model}"
        
        if not os.path.exists(model_path):
            print(f"{model} is not found")
            raise NotImplementedError
        else:
            print(f"{args.m}--{ep}")
        
        data = HierDataset(data_path=data_path, create=False, version=args.dv)
        checkpoint = torch.load(model_path, map_location='cpu')
        embedding = checkpoint['child_embedding']

        print(model)
        if 'UniTRep' in args.m:
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
            
            auc_list, ap_list, acc_list = super_diachornic_link_pred(embedding, data.child_link, model, num_rels=4, mode="test", k=5, max_iter=args.i)
        else:
            auc_list, ap_list, acc_list = super_static_link_pred(embedding, data.child_link, num_rels=4, mode="test", k=5, max_iter=args.i)
        
        print("Link Prediction -- AUC | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*auc_list))
        print("Link Prediction -- AP | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*ap_list))
        print("Link Prediction -- ACC | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*acc_list))

        res_auc.append(auc_list[-1])
        res_ap.append(ap_list[-1])
        res_acc.append(acc_list[-1])
    
    print(np.mean(res_auc), np.std(res_auc))
    print(np.mean(res_ap), np.std(res_ap))
    print(np.mean(res_acc), np.std(res_acc))
    

# nohup python talent_flow_infer.py -v 4 -m UniTRep > log/talent_flow_inference/UniTRep4 &
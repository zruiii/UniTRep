import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import random
import argparse
import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
from dataset import HierDataset
from dataloader import MergeDataset
from evaluate import temporal_link_pred, temporal_recommend, temporal_traj_pred
from utils import sbatch_convert, gbatch_convert


def setup_seed(seed):
    """ Setup Random Seed """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ntypes", type=int, default=2)
    parser.add_argument("--num_rels", type=int, default=4)
    parser.add_argument("--bs", type=int, default=2000)
    parser.add_argument("--cuda", type=int, default=3)
    parser.add_argument("--model", default='UniTRep_HE')
    parser.add_argument("--sub_v", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--t2v_dim", type=int, default=64)
    parser.add_argument("--edge_dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.5)             # 0.1  0.3  0.5  0.7 0.9 
    parser.add_argument("--num_heads", type=int, default=4)             # hyper-parameter
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--neg_num", type=int, default=10)
    parser.add_argument("--contra_num", type=int, default=200)
    parser.add_argument("--mj_pool", type=str, default="attn")
    parser.add_argument("--traj_pool", type=str, default="attn")
    parser.add_argument("--seed", type=int, default=2023)               # 36, 42, 17, 24, 2023, 50, 66, 74, 81, 98
    parser.add_argument("--lr", type=float, default=1e-4)               # 1e-3, 1e-4, 1e-5
    parser.add_argument("--decay", type=float, default=1e-4)            # 1e-4, 1e-3, 5e-4
    parser.add_argument("--lam_tm", type=float, default=0.3)            # 0.1, 0.3, 0.5, 0.7, 0.9 (参敏实验)
    parser.add_argument("--propor_mj", type=int, default=5)             # 1, 3, 5, 7, 9 
    parser.add_argument("--seq_loss", type=str, default="contra")
    parser.add_argument("--save", action='store_true')      # save model
    parser.add_argument("--log", action='store_true')   # tensorboard
    parser.add_argument("--dv", type=int, default=5)    # data version
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.cuda))
    setup_seed(2022) 

    ######## Data & Model ########
    log_dir = os.path.join(os.getcwd(), "log/tensorboard")
    data_path = "/data/data_v5/"
    model_save_path = os.path.join(os.getcwd(), "model_save", f"{args.model}_{args.sub_v}")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    data = HierDataset(data_path=data_path, version=args.dv, create=False)
    train_parent_g = data.train_parent_g
    train_child_g = data.train_child_g
    train_seq = data.train_seq

    ds = MergeDataset(train_parent_g,
                      train_child_g,
                      train_seq,
                      data.pool_map,
                      neg_num=args.neg_num,
                      batch_size=args.bs)
    dataloader = DataLoader(ds, num_workers=4, prefetch_factor=2, pin_memory=True)
    
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
                traj_pool=args.traj_pool,
                device=device).to(device)
    print(model)
    
    if args.log: writer = SummaryWriter(log_dir=log_dir + f'/{args.model}_{args.sub_v}' + '/')

    ######## Traning & Testing ########
    # Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print("Optimizer: Adam")
    
    for ep in range(args.epochs):
        total_seq_loss, total_child_graph_loss, total_parent_graph_loss = 0, 0, 0
        
        # train
        for it, (g_batch, s_batch) in enumerate(tqdm.tqdm(dataloader, desc="Training")):
            g_batch = gbatch_convert(g_batch)
            s_batch = sbatch_convert(s_batch)
            
            model.train()
            seq_embed, child_embed, parent_embed = model(g_batch, s_batch)
            seq_loss = model.get_seq_loss(seq_embed, child_embed, s_batch, k=args.contra_num)
            parent_graph_loss = model.get_graph_loss(embed=parent_embed,
                                                     nodes=g_batch['parent_nodes'],
                                                     edges=g_batch['parent_edges'],
                                                     labels=g_batch['parent_labels'].to(device),
                                                     neg_num=args.neg_num,
                                                     ntype='parent')
                        
            child_graph_loss = model.get_graph_loss(embed=child_embed,
                                                    nodes=g_batch['child_nodes'],
                                                    edges=g_batch['child_edges'],
                                                    labels=g_batch['child_labels'].to(device),
                                                    neg_num=args.neg_num,
                                                    ntype='child')
       
            lam_uc = (1 - args.lam_tm) / (args.propor_mj + 1)
            lam_mj = 1 - args.lam_tm - lam_uc
            loss = seq_loss * args.lam_tm + child_graph_loss * lam_mj + parent_graph_loss * lam_uc
            
            total_seq_loss += seq_loss.item()
            total_child_graph_loss += child_graph_loss.item()
            total_parent_graph_loss += parent_graph_loss.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip gradients
            optimizer.step()
            
            if args.log: writer.add_scalar("Loss/Learning Rate", optimizer.param_groups[0]["lr"], ep * len(dataloader) + it + 1)
        
        print("Epoch-{:03d}  M&J Graph Loss: {:.4f}  S&C Graph Loss: {:.4f}  Sequence Loss: {:.4f}".format(
            ep + 1, total_child_graph_loss / (it + 1), total_parent_graph_loss / (it + 1),
            total_seq_loss / (it + 1)))
            
        # eval
        if (ep + 1) % 5 == 0:
            model.eval()
            if args.log:
                writer.add_scalar("Loss/M&J Graph Loss", total_child_graph_loss / (it + 1), ep + 1)
                writer.add_scalar("Loss/S&C Graph Loss", total_parent_graph_loss / (it + 1), ep + 1)
                writer.add_scalar("Loss/Sequence Loss", total_seq_loss / (it + 1), ep + 1)
                
            child_embed, parent_embed, pre_child_embed = model.inference(g=train_child_g, c2p_map=data.pool_map, 
                                                                         g_bs=10000, sequence=train_seq, s_bs=2000)
            
            if args.save and (ep + 1) % 5 == 0: 
                torch.save({'state_dict': model.state_dict(), 'epoch': ep + 1, 'args': args, 'child_embedding': child_embed.cpu(),
                            'parent_embedding': parent_embed.cpu(), 'pre_child_embed': pre_child_embed.cpu()}, model_save_path + f"/{args.model}_{ep+1}.pth")
            
            ################ Major & Job ##############
            auc_list, ap_list, acc_list, lp_loss = temporal_link_pred(model, child_embed, data.child_link, num_rels=args.num_rels, ntype='child', mode='test')
            ndcg_10, ndcg_50, recall_10, recall_50, traj_loss = temporal_traj_pred(model, train_child_g, pre_child_embed, child_embed, data.traj_pred, 
                                                                            device, seq_loss=args.seq_loss)
                
            print('*' * 10, "Major & Job Evaluation", '*' * 10)
            print("Link Prediction -- AUC | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*auc_list))
            print("Link Prediction -- AP | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*ap_list))
            print("Link Prediction -- ACC | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*acc_list))
            
            print("Recommendation -- MRR | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*mrr_list))
            print("Recommendation -- Hits@1 | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*hits1_list))
            print("Recommendation -- Hits@3 | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*hits3_list))
            print("Recommendation -- Hits@10 | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*hits10_list))
            
            print("Trajectory Prediction -- NDCG@10: {:.4f};  NDCG@50: {:.4f};  Recall@10: {:.4f};  Recall@50: {:.4f};  average: {:.4f}".format(
                ndcg_10, ndcg_50, recall_10, recall_50, np.mean([ndcg_10, ndcg_50, recall_10, recall_50])
            ))
            
            if args.log:
                writer.add_scalar("Loss/lp_loss", lp_loss, ep + 1)          # testing link prediction BCE loss
                writer.add_scalar("M&J Link Prediction/AUC average", auc_list[-1], ep + 1)
                writer.add_scalar("M&J Link Prediction/AP average", ap_list[-1], ep + 1)
                writer.add_scalar("M&J Link Prediction/ACC average", acc_list[-1], ep + 1)
                
                writer.add_scalar("M&J Recommendation/MRR", mrr_list[-1], ep + 1)
                writer.add_scalar("M&J Recommendation/Hits@1", hits1_list[-1], ep + 1)
                writer.add_scalar("M&J Recommendation/Hits@3", hits3_list[-1], ep + 1)
                writer.add_scalar("M&J Recommendation/Hits@10", hits10_list[-1], ep + 1)
                
                writer.add_scalar("Trajectory Prediction/NDCG@10", ndcg_10, ep+1)
                writer.add_scalar("Trajectory Prediction/NDCG@50", ndcg_50, ep+1)
                writer.add_scalar("Trajectory Prediction/Recall@10", recall_10, ep+1)
                writer.add_scalar("Trajectory Prediction/Recall@50", recall_50, ep+1)
                writer.add_scalar("Trajectory Prediction/traj_loss", traj_loss, ep+1)

            ################ School & Company ##############
            auc_list, ap_list, acc_list, lp_loss = temporal_link_pred(model, parent_embed, data.parent_link, num_rels=args.num_rels, ntype='parent', mode='test')
            mrr_list, hits1_list, hits3_list, hits10_list = temporal_recommend(model, parent_embed, data.parent_rank, num_rels=args.num_rels, ntype='parent')

            print('*' * 10, "School & Company Evaluation", '*' * 10)
            print("Link Prediction -- AUC | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*auc_list))
            print("Link Prediction -- AP | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*ap_list))
            print("Link Prediction -- ACC | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*acc_list))
            
            print("Recommendation -- MRR | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*mrr_list))
            print("Recommendation -- Hits@1 | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*hits1_list))
            print("Recommendation -- Hits@3 | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*hits3_list))
            print("Recommendation -- Hits@10 | eType-0: {:.4f};  eType-1: {:.4f};  eType-2: {:.4f};  eType-3: {:.4f};  average: {:.4f}".format(*hits10_list))
            print('\n')
            
            if args.log:
                writer.add_scalar("Loss/lp_loss_parent", lp_loss, ep + 1)          # testing link prediction BCE loss
                writer.add_scalar("S&C Link Prediction/AUC average", auc_list[-1], ep + 1)
                writer.add_scalar("S&C Link Prediction/AP average", ap_list[-1], ep + 1)
                writer.add_scalar("S&C Link Prediction/ACC average", acc_list[-1], ep + 1)
                            
                writer.add_scalar("S&C Recommendation/MRR", mrr_list[-1], ep + 1)
                writer.add_scalar("S&C Recommendation/Hits@1", hits1_list[-1], ep + 1)
                writer.add_scalar("S&C Recommendation/Hits@3", hits3_list[-1], ep + 1)
                writer.add_scalar("S&C Recommendation/Hits@10", hits10_list[-1], ep + 1)
            
            
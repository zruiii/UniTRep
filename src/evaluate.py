
import tqdm
import math
from statistics import mean
import dgl
import torch
import numpy as np
import torch.nn.functional as F
from dgl.sampling import sample_neighbors
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import graph_to_sequence_block, sbatch_convert
from sklearn.metrics import roc_auc_score, accuracy_score, ndcg_score, average_precision_score

from dataloader import ValSeqDataset

# ************ Using the resulted representations and proposed framework directly ************

@torch.no_grad()
def static_link_pred(model, embed, data, num_rels=4, ntype='child', mode='test'):
    """ Node link prediction under statistic scenario.

    Parameters
    ----------
    model
        _description_
    embed
        _description_
    data
        _description_
    num_rels, optional
        _description_, by default 4
    ntype, optional
        _description_, by default 'child'
    mode, optional
        _description_, by default 'test'

    Returns
    -------
        _description_
    """
    auc_list, ap_list = [], []
    score_list, pred_list, label_list, acc_list = [], [], [], []
    
    for rel in range(num_rels):
        pos_edges = data[rel][f'{mode}_pos_edges']      # [E, 3]
        neg_edges = data[rel][f'{mode}_neg_edges']      # [E, 3]
        src = torch.LongTensor(np.concatenate((pos_edges[:, 0], neg_edges[:, 0]), axis=0))
        dst = torch.LongTensor(np.concatenate((pos_edges[:, 1], neg_edges[:, 1]), axis=0))
        labels = np.concatenate((np.ones(pos_edges.shape[0]), np.zeros(neg_edges.shape[0])))
        
        src_embed = embed[src]
        dst_embed = embed[dst]
        if ntype == 'child':
            score = model.child_predictors[rel](torch.cat((src_embed, dst_embed), dim=1)).flatten()
        else:
            score = model.parent_predictors[rel](torch.cat((src_embed, dst_embed), dim=1)).flatten()
            
        score = score.cpu().numpy()
        auc = roc_auc_score(labels, score)
        ap = average_precision_score(labels, score)
        
        auc_list.append(auc)
        ap_list.append(ap)
        score_list.extend(list(score))
        label_list.extend(list(labels))
        
        # NOTE: 选取概率最大的一半边预测为正样本，其余预测为负样本来计算 ACC, 没有绝对的阈值
        median_score = np.median(score)
        pred = np.zeros(len(score))
        pred[score > median_score] = 1
        acc = accuracy_score(labels, pred)
        acc_list.append(acc)
        pred_list.extend(pred)
        
        # auc_list.append(auc)
        # acc_list.append(acc)
        # score_list.extend(list(score))
        # pred_list.extend(pred)
        # label_list.extend(list(labels))
        
    auc_list.append(roc_auc_score(label_list, score_list)) 
    ap_list.append(average_precision_score(label_list, score_list))
    acc_list.append(accuracy_score(label_list, pred_list))
    return auc_list, ap_list, acc_list
    
    # score_list = torch.sigmoid(torch.tensor(score_list)).numpy()
    # print(roc_auc_score(label_list, score_list.round()))
    # print(acc_list[-1])
    # return auc_list , acc_list


@torch.no_grad()
def static_recommend(model, embed, data, num_rels=4, ntype='child', mode="test", batch_size=5000, graph="hete"):
    """ Node-Level Recommendation under statistic scenario.

    Parameters
    ----------
    model
        _description_
    embed
        _description_
    data
        _description_
    num_rels, optional
        _description_, by default 4
    ntype, optional
        _description_, by default 'child'
    mode, optional
        _description_, by default "test"
    batch_size, optional
        _description_, by default 5000

    Returns
    -------
        _description_
    """
    mrr_list, hits1_list, hits3_list, hits10_list = [], [], [], []
    
    for rel in range(num_rels):
        src = torch.LongTensor(data[rel][f'{mode}_src'][:, 0])
        rr = torch.zeros(src.shape[0])
        hits1, hits3, hits10 = torch.zeros(src.shape[0]), torch.zeros(src.shape[0]), torch.zeros(src.shape[0])
        for start in range(0, src.shape[0], batch_size):
            end = min(start + batch_size, src.shape[0])     
            pos_dst = torch.LongTensor(data[rel][f'{mode}_dst'][start:end])            # (B, )
            neg_dst = torch.LongTensor(data[rel][f'{mode}_neg_dst'][start:end, :200])        # (B, k)   # NOTE: k negative candidates
            all_dst = torch.cat((pos_dst[:, None], neg_dst), dim=1)                          # (B, k+1)
            all_src = src[start:end][:, None].repeat(1, all_dst.shape[1])                    # (B, k+1)
               
            src_embed = embed[all_src]       # (B, k+1, H)
            dst_embed = embed[all_dst]
            if graph == 'hete':
                if ntype == 'child':
                    pred = model.child_predictors[rel](torch.cat((src_embed, dst_embed), dim=2)).squeeze(-1)
                else:
                    pred = model.parent_predictors[rel](torch.cat((src_embed, dst_embed), dim=2)).squeeze(-1)
            elif graph == 'homo':
                if ntype == 'child':
                    pred = model.child_predictor(torch.cat((src_embed, dst_embed), dim=2)).squeeze(-1)
                else:
                    pred = model.parent_predictor(torch.cat((src_embed, dst_embed), dim=2)).squeeze(-1)
            else:
                raise NotImplementedError
            
            y_pred_pos, y_pred_neg = pred[:, 0].view(-1, 1), pred[:, 1:]
            optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)  # 概率：负样本 ≥ 正样本
            pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)  # 概率：负样本 > 正样本
            ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1  # 正样本排名
            hits1[start:end] = (ranking_list <= 1).to(torch.float)
            hits3[start:end] = (ranking_list <= 3).to(torch.float)
            hits10[start:end] = (ranking_list <= 10).to(torch.float)
            rr[start:end] = 1. / ranking_list.to(torch.float)
        
        mrr_list.append(rr.mean().item())
        hits1_list.append(hits1.mean().item())
        hits3_list.append(hits3.mean().item())
        hits10_list.append(hits10.mean().item())
    
    counts = [data[rel][f'{mode}_src'].shape[0] for rel in range(num_rels)]
    mrr_ave = sum([x[0] * x[1] for x in list(zip(counts, mrr_list))]) / sum(counts)
    hits1_ave = sum([x[0] * x[1] for x in list(zip(counts, hits1_list))]) / sum(counts)
    hits3_ave = sum([x[0] * x[1] for x in list(zip(counts, hits3_list))]) / sum(counts)
    hits10_ave = sum([x[0] * x[1] for x in list(zip(counts, hits10_list))]) / sum(counts)
    
    mrr_list.append(mrr_ave)
    hits1_list.append(hits1_ave)
    hits3_list.append(hits3_ave)
    hits10_list.append(hits10_ave)
     
    return mrr_list, hits1_list, hits3_list, hits10_list


@torch.no_grad()
def static_traj_pred(model, embed, data, device, mode="test", bs=2000, seq_loss="contra"):
    """ Trajectory-Node Mathching under Stastic Scenario. This version follow the Graph-to-Sequence Framework.
        It means that we use the node embedding of Graph Module to match the sequence embedding.

    Parameters
    ----------
    model
        Input Model
    embed
        Node Embedding
    data
        Data for evaluation
    device
        _description_
    mode, optional
        testing for validation dataset, by default "test"
    bs, optional
        batch size for testing sequences, by default 2000
    seq_loss, optional
        the method for calculating sequence loss, by default "contra"

    Returns
    -------
        Recommendation metrics, including NDCG@N and Recall@N. And the matching loss on test data.
    """
    full_seq = data[f'{mode}_seq']
    full_dst = data[f'{mode}_dst']
    full_neg_dst = data[f'{mode}_neg_dst']
    
    ndcg_10, ndcg_50, recall_10, recall_50 = [], [], [], []
    batch_num = []
    
    score_all = []
    all_index = np.arange(len(full_seq))
    num_batch = math.ceil(len(full_seq) / bs)
    for idx in range(num_batch):
        if idx != num_batch - 1:
            _index = all_index[idx * bs: (idx + 1) * bs]
        else:
            _index = all_index[idx * bs: len(full_seq)]
        
        ############ Get Sequence Batch ############
        batch_seq = [full_seq[i] for i in _index]
        batch_dst = [full_dst[i] for i in _index]
        batch_neg_dst = [full_neg_dst[i] for i in _index]
        
        seq_len = list(map(len, batch_seq))
        seq = np.zeros((len(_index), max(seq_len)))         # nids
        dur = np.zeros((len(_index), max(seq_len)))         # duration
        stime = np.zeros((len(_index), max(seq_len)))       # start time
        etime = np.zeros((len(_index), max(seq_len)))       # end time
        
        for it, (sequence, length) in enumerate(zip(batch_seq, seq_len)):
            seq[it][:length] = [x[0] for x in sequence]
            stime[it][:length] = [x[1] for x in sequence]
            etime[it][:length] = [x[2] for x in sequence]
        
        # resorted
        sorted_id = sorted(range(len(seq_len)), key=lambda k: seq_len[k], reverse=True)
        seq_len = np.array([seq_len[idx] for idx in sorted_id])
        seq = torch.LongTensor(seq[sorted_id])
        batch_dst = torch.LongTensor(batch_dst)[sorted_id]
        batch_neg_dst = torch.LongTensor(batch_neg_dst)[sorted_id]
        stime = torch.LongTensor(stime[sorted_id]).to(device)
        etime = torch.LongTensor(etime[sorted_id]).to(device)
        dur = etime - stime
        
        ############ Sequence Modeling ############
        seq_embed = embed[seq]
        time_embed = model.rel_time(dur)
        h0, c0 = model.init_hidden(seq.shape[0])
        packed_input = pack_padded_sequence(torch.cat((seq_embed, time_embed), dim=-1), seq_len, batch_first=True)
        packed_output, (_, _) = model.lstm(packed_input, (h0, c0))
        seq_embed, _ = pad_packed_sequence(packed_output, batch_first=True)  # [B, L, H]
        
        ############ Matching ############
        t = torch.LongTensor(seq_len-1)[:, None].to(device)
        history_embed = seq_embed.gather(1, t.unsqueeze(-1).repeat(1, 1, 128))        # [B, H]
        all_dst = torch.cat((batch_dst[:, None], batch_neg_dst), dim=1)
        dst_embed = embed[all_dst]                # [B, k+1, H]
        
        score = F.cosine_similarity(history_embed.repeat(1, dst_embed.shape[1], 1), dst_embed, dim=-1)     # [B, k+1]
        labels = torch.cat((torch.ones((score.shape[0], 1)), torch.zeros((score.shape[0], score.shape[1]-1))), dim=1)
        
        ############ Calculate Metrics ############
        ndcg_10.append(ndcg_score(labels.cpu().numpy(), score.cpu().numpy(), k=10))
        ndcg_50.append(ndcg_score(labels.cpu().numpy(), score.cpu().numpy(), k=50))
        
        sort_score = torch.sort(score, dim=1, descending=True).values.cpu().numpy()
        indices = torch.sort(score, dim=1, descending=True).indices
        sort_labels = np.zeros(sort_score.shape)
        sort_labels[(indices == 0).cpu().numpy()] = 1
        
        recall_10.append(sort_labels[:, :10].sum() / sort_labels.shape[0])
        recall_50.append(sort_labels[:, :50].sum() / sort_labels.shape[0])
        
        batch_num.append(_index.shape[0])
        score_all.append(score)
        
    # loss
    score = torch.cat(score_all, dim=0)
    if seq_loss == "contra":
        loss = torch.exp(score[:, 0] / 0.8) / torch.exp(score[:, 1:] / 0.8).sum(1)
    elif seq_loss == "bpr":
        loss = torch.sigmoid(score[:, 0] * (score.shape[1] - 1) - score[:, 1:].sum(1))
    loss = -torch.log(loss).mean()
    
    ndcg_10 = sum([x[0] * x[1] for x in zip(ndcg_10, batch_num)]) / sum(batch_num)
    ndcg_50 = sum([x[0] * x[1] for x in zip(ndcg_50, batch_num)]) / sum(batch_num)
    recall_10 = sum([x[0] * x[1] for x in zip(recall_10, batch_num)]) / sum(batch_num)
    recall_50 = sum([x[0] * x[1] for x in zip(recall_50, batch_num)]) / sum(batch_num)
        
    return ndcg_10, ndcg_50, recall_10, recall_50, loss.item()


@torch.no_grad()
def static_traj_pred2(model, pre_embed, embed, data, device, mode="test", bs=2000, seq_loss="contra"):
    """ Trajectory-Node Mathching under Stastic Scenario. This version follow the Graph-to-Sequence-to Graph
        Framework. It means that we use the sequence-enhanced node embedding to match with the sequence embedding.

    Parameters
    ----------
    model
        Input Model
    pre_embed
        The first round node embedding
    embed
        The final output node embedding
    data
        Data for evaluation
    device
        _description_
    mode, optional
        testing for validation dataset, by default "test"
    bs, optional
        batch size for testing sequences, by default 2000
    seq_loss, optional
        the method for calculating sequence loss, by default "contra"

    Returns
    -------
        Recommendation metrics, including NDCG@N and Recall@N. And the matching loss on test data.
    """
    full_seq = data[f'{mode}_seq']
    full_dst = data[f'{mode}_dst']
    full_neg_dst = data[f'{mode}_neg_dst']
    
    ndcg_10, ndcg_50, recall_10, recall_50 = [], [], [], []
    batch_num = []
    
    score_all = []
    all_index = np.arange(len(full_seq))
    num_batch = math.ceil(len(full_seq) / bs)
    for idx in range(num_batch):
        if idx != num_batch - 1:
            _index = all_index[idx * bs: (idx + 1) * bs]
        else:
            _index = all_index[idx * bs: len(full_seq)]
        
        ############ Get Sequence Batch ############
        batch_seq = [full_seq[i] for i in _index]
        batch_dst = [full_dst[i] for i in _index]
        batch_neg_dst = [full_neg_dst[i] for i in _index]
        
        seq_len = list(map(len, batch_seq))
        seq = np.zeros((len(_index), max(seq_len)))         # nids
        dur = np.zeros((len(_index), max(seq_len)))         # duration
        stime = np.zeros((len(_index), max(seq_len)))       # start time
        etime = np.zeros((len(_index), max(seq_len)))       # end time
        
        for it, (sequence, length) in enumerate(zip(batch_seq, seq_len)):
            seq[it][:length] = [x[0] for x in sequence]
            stime[it][:length] = [x[1] for x in sequence]
            etime[it][:length] = [x[2] for x in sequence]
        
        # resorted
        sorted_id = sorted(range(len(seq_len)), key=lambda k: seq_len[k], reverse=True)
        seq_len = np.array([seq_len[idx] for idx in sorted_id])
        seq = torch.LongTensor(seq[sorted_id])
        stime = torch.LongTensor(stime[sorted_id]).to(device)
        etime = torch.LongTensor(etime[sorted_id]).to(device)
        dur = etime - stime
        
        ############ Sequence Modeling ############
        seq_embed = pre_embed[seq]
        time_embed = model.rel_time(dur)
        h0, c0 = model.init_hidden(seq.shape[0])
        packed_input = pack_padded_sequence(torch.cat((seq_embed, time_embed), dim=-1), seq_len, batch_first=True)
        packed_output, (_, _) = model.lstm(packed_input, (h0, c0))
        seq_embed, _ = pad_packed_sequence(packed_output, batch_first=True)  # [B, L, H]
        
        ############ Matching ############
        t = torch.LongTensor(seq_len-1)[:, None].to(device)
        history_embed = seq_embed.gather(1, t.unsqueeze(-1).repeat(1, 1, 128))        # [B, 1, H]
        all_dst = torch.cat((torch.LongTensor(batch_dst)[:, None], torch.LongTensor(batch_neg_dst)), dim=1)
        dst_embed = embed[all_dst]                # [B, k+1, H]
        
        score = F.cosine_similarity(history_embed.repeat(1, dst_embed.shape[1], 1), dst_embed, dim=-1)     # [B, k+1]
        labels = torch.cat((torch.ones((score.shape[0], 1)), torch.zeros((score.shape[0], score.shape[1]-1))), dim=1)
        
        ############ Calculate Metrics ############
        ndcg_10.append(ndcg_score(labels.cpu().numpy(), score.cpu().numpy(), k=10))
        ndcg_50.append(ndcg_score(labels.cpu().numpy(), score.cpu().numpy(), k=50))
        
        sort_score = torch.sort(score, dim=1, descending=True).values.cpu().numpy()
        indices = torch.sort(score, dim=1, descending=True).indices
        sort_labels = np.zeros(sort_score.shape)
        sort_labels[(indices == 0).cpu().numpy()] = 1
        
        recall_10.append(sort_labels[:, :10].sum() / sort_labels.shape[0])
        recall_50.append(sort_labels[:, :50].sum() / sort_labels.shape[0])
        
        batch_num.append(_index.shape[0])
        score_all.append(score)
    
    # loss
    score = torch.cat(score_all, dim=0)
    if seq_loss == "contra":
        loss = torch.exp(score[:, 0] / 0.8) / torch.exp(score[:, 1:] / 0.8).sum(1)
    elif seq_loss == "bpr":
        loss = torch.sigmoid(score[:, 0] * (score.shape[1] - 1) - score[:, 1:].sum(1))
    loss = -torch.log(loss).mean()
    
    ndcg_10 = sum([x[0] * x[1] for x in zip(ndcg_10, batch_num)]) / sum(batch_num)
    ndcg_50 = sum([x[0] * x[1] for x in zip(ndcg_50, batch_num)]) / sum(batch_num)
    recall_10 = sum([x[0] * x[1] for x in zip(recall_10, batch_num)]) / sum(batch_num)
    recall_50 = sum([x[0] * x[1] for x in zip(recall_50, batch_num)]) / sum(batch_num)
        
    return ndcg_10, ndcg_50, recall_10, recall_50, loss.item()


@torch.no_grad()
def temporal_link_pred(model, embed, data, num_rels=4, ntype='child', mode='test'):
    # time-aware
    device = model.device
    auc_list, ap_list, acc_list = [], [], []
    score_list, pred_list, label_list = [], [], []
    for rel in range(num_rels):
        pos_edges = data[rel][f'{mode}_pos_edges']      # [E, 3]
        neg_edges = data[rel][f'{mode}_neg_edges']      # [E, 3]
        src = torch.LongTensor(np.concatenate((pos_edges[:, 0], neg_edges[:, 0]), axis=0))
        dst = torch.LongTensor(np.concatenate((pos_edges[:, 1], neg_edges[:, 1]), axis=0))
        t = torch.LongTensor(np.concatenate((pos_edges[:, 2], neg_edges[:, 2]), axis=0))[:, None].to(device)
        labels = np.concatenate((np.ones(pos_edges.shape[0]), np.zeros(neg_edges.shape[0])))
        
        src_embed = embed[src]
        dst_embed = embed[dst]
        if ntype == 'child':
            src_embed = model.child_dia(src, src_embed, t)
            dst_embed = model.child_dia(dst, dst_embed, t)
            score = model.child_predictors[rel](torch.cat((src_embed, dst_embed), dim=1)).flatten()
        else:
            src_embed = model.parent_dia(src, src_embed, t)
            dst_embed = model.parent_dia(dst, dst_embed, t)
            score = model.parent_predictors[rel](torch.cat((src_embed, dst_embed), dim=1)).flatten()
            
        score = score.cpu().numpy()
        auc = roc_auc_score(labels, score)
        ap = average_precision_score(labels, score)
        
        auc_list.append(auc)
        ap_list.append(ap)
        score_list.extend(list(score))
        label_list.extend(list(labels))
        
        # NOTE: 选取概率最大的一半边预测为正样本，其余预测为负样本来计算 ACC, 没有绝对的阈值
        median_score = np.median(score)
        pred = np.zeros(len(score))
        pred[score > median_score] = 1
        acc = accuracy_score(labels, pred)
        acc_list.append(acc)
        pred_list.extend(pred)
        
        # auc_list.append(auc)
        # acc_list.append(acc)
        # score_list.extend(list(score))
        # pred_list.extend(pred)
        # label_list.extend(list(labels))
    
    loss = F.binary_cross_entropy_with_logits(torch.tensor(score_list), torch.tensor(label_list))
    auc_list.append(roc_auc_score(label_list, score_list))
    ap_list.append(average_precision_score(label_list, score_list))
    acc_list.append(accuracy_score(label_list, pred_list))
    return auc_list, ap_list, acc_list, loss


@torch.no_grad()
def temporal_recommend(model, embed, data, num_rels=4, ntype='child', mode="test", batch_size=2000):
    device = model.device
    mrr_list, hits1_list, hits3_list, hits10_list = [], [], [], []
    
    for rel in range(num_rels):
        src = torch.LongTensor(data[rel][f'{mode}_src'][:, 0])
        t = torch.LongTensor(data[rel][f'{mode}_src'][:, 1])
        rr = torch.zeros(src.shape[0])
        hits1, hits3, hits10 = torch.zeros(src.shape[0]), torch.zeros(src.shape[0]), torch.zeros(src.shape[0])
        for start in range(0, src.shape[0], batch_size):
            end = min(start + batch_size, src.shape[0])     
            pos_dst = torch.LongTensor(data[rel][f'{mode}_dst'][start:end])            # (B, )
            neg_dst = torch.LongTensor(data[rel][f'{mode}_neg_dst'][start:end, :])        # (B, k)   # NOTE: k negative candidates
            all_dst = torch.cat((pos_dst[:, None], neg_dst), dim=1)                          # (B, k+1)
            all_src = src[start:end][:, None].repeat(1, all_dst.shape[1])                    # (B, k+1)
            t_batch = t[start:end][:, None].repeat(1, all_dst.shape[1]).unsqueeze(-1).to(device)        # (B, k+1, 1)
               
            h_src = embed[all_src]       # (B, k+1, H)
            h_dst = embed[all_dst]
            if ntype == 'child':
                src_embed = model.child_dia(all_src, h_src, t_batch)
                dst_embed = model.child_dia(all_dst, h_dst, t_batch)
                pred = model.child_predictors[rel](torch.cat((src_embed, dst_embed), dim=-1)).squeeze(-1)
            else:
                src_embed = model.parent_dia(all_src, h_src, t_batch)
                dst_embed = model.parent_dia(all_dst, h_dst, t_batch)
                pred = model.parent_predictors[rel](torch.cat((src_embed, dst_embed), dim=-1)).squeeze(-1)
        
            y_pred_pos, y_pred_neg = pred[:, 0].view(-1, 1), pred[:, 1:]
            optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)  # 概率：负样本 ≥ 正样本
            pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)  # 概率：负样本 > 正样本
            ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1  # 正样本排名
            hits1[start:end] = (ranking_list <= 1).to(torch.float)
            hits3[start:end] = (ranking_list <= 3).to(torch.float)
            hits10[start:end] = (ranking_list <= 10).to(torch.float)
            rr[start:end] = 1. / ranking_list.to(torch.float)
        
        mrr_list.append(rr.mean().item())
        hits1_list.append(hits1.mean().item())
        hits3_list.append(hits3.mean().item())
        hits10_list.append(hits10.mean().item())
    
    counts = [data[rel][f'{mode}_src'].shape[0] for rel in range(num_rels)]
    mrr_ave = sum([x[0] * x[1] for x in list(zip(counts, mrr_list))]) / sum(counts)
    hits1_ave = sum([x[0] * x[1] for x in list(zip(counts, hits1_list))]) / sum(counts)
    hits3_ave = sum([x[0] * x[1] for x in list(zip(counts, hits3_list))]) / sum(counts)
    hits10_ave = sum([x[0] * x[1] for x in list(zip(counts, hits10_list))]) / sum(counts)
    
    mrr_list.append(mrr_ave)
    hits1_list.append(hits1_ave)
    hits3_list.append(hits3_ave)
    hits10_list.append(hits10_ave)
     
    return mrr_list, hits1_list, hits3_list, hits10_list


@torch.no_grad()
def temporal_traj_pred(model, g, pre_embed, final_embed, data, device, mode="test", bs=2000, seq_loss="contra"):
    batch_num, score_all = [], []
    ndcg_10, ndcg_50, recall_10, recall_50 = [], [], [], []
    s_dataloader = torch.utils.data.DataLoader(ValSeqDataset(g, data, batch_size=bs, mode=mode), 
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
        batch_neg_dst = torch.LongTensor(s_batch['neg_dst']).to(device)
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
        
        ############ Matching ############
        last_index = torch.LongTensor(seq_len-1)[:, None].to(device)
        seq_embed = seq_embed.gather(1, last_index.unsqueeze(-1).repeat(1, 1, 128))        # [B, H]
        all_dst = torch.cat((batch_dst[:, None], batch_neg_dst), dim=1)
        dst_embed = final_embed[all_dst]                # [B, k+1, H]
        
        etime = etime.gather(1, last_index).repeat(1, dst_embed.shape[1]).unsqueeze(-1)
        dst_embed = model.child_dia(all_dst, dst_embed, etime)
        score = F.cosine_similarity(seq_embed.repeat(1, dst_embed.shape[1], 1), dst_embed, dim=-1)     # [B, k+1]
        labels = torch.cat((torch.ones((score.shape[0], 1)), torch.zeros((score.shape[0], score.shape[1]-1))), dim=1)
        
        ############ Calculate Metrics ############
        ndcg_10.append(ndcg_score(labels.cpu().numpy(), score.cpu().numpy(), k=10))
        ndcg_50.append(ndcg_score(labels.cpu().numpy(), score.cpu().numpy(), k=50))
        
        sort_score = torch.sort(score, dim=1, descending=True).values.cpu().numpy()
        indices = torch.sort(score, dim=1, descending=True).indices
        sort_labels = np.zeros(sort_score.shape)
        sort_labels[(indices == 0).cpu().numpy()] = 1
        
        recall_10.append(sort_labels[:, :10].sum() / sort_labels.shape[0])
        recall_50.append(sort_labels[:, :50].sum() / sort_labels.shape[0])
        
        batch_num.append(score.shape[0])
        score_all.append(score)
    
    # loss
    score = torch.cat(score_all, dim=0)
    if seq_loss == "contra":
        loss = torch.exp(score[:, 0] / 0.8) / torch.exp(score[:, 1:] / 0.8).sum(1)
    elif seq_loss == "bpr":
        loss = torch.sigmoid(score[:, 0] * (score.shape[1] - 1) - score[:, 1:].sum(1))
    loss = -torch.log(loss).mean()
    
    ndcg_10 = sum([x[0] * x[1] for x in zip(ndcg_10, batch_num)]) / sum(batch_num)
    ndcg_50 = sum([x[0] * x[1] for x in zip(ndcg_50, batch_num)]) / sum(batch_num)
    recall_10 = sum([x[0] * x[1] for x in zip(recall_10, batch_num)]) / sum(batch_num)
    recall_50 = sum([x[0] * x[1] for x in zip(recall_50, batch_num)]) / sum(batch_num)
        
    return ndcg_10, ndcg_50, recall_10, recall_50, loss.item()







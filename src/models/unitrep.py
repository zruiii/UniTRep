import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import dgl
import tqdm
from dgl.nn import TypedLinear, EdgeWeightNorm
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader

from dataloader import SeqDataset
from utils import negtive_sampler, sbatch_convert
from .HTGT import HTGTLayer
from .modules import Time2Vec, Diachronic, HeteLinear


class UniTRep(nn.Module):
    def __init__(self,
                 parent_num,
                 major_num,
                 job_num,
                 h_dim,
                 e_dim,
                 t2v_dim,
                 num_rels,
                 num_ntypes,
                 num_heads,
                 device,
                 rnn_layer=1,
                 mj_pool="avg",
                 traj_pool="attn",
                 gamma=0.5,
                 activ="sin"):
        
        super(UniTRep, self).__init__()
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.num_ntypes = num_ntypes
        self.rnn_layer = rnn_layer
        self.device = device
        self.mj_pool = mj_pool
        self.traj_pool = traj_pool
        self.norm = EdgeWeightNorm(norm='right')
        
        # init graph embedding, memory cost
        child_num = major_num + job_num + 1
        self.embedding = nn.Embedding(child_num, h_dim)
        
        history_embedding = self.embedding.weight.clone().detach()
        self.history_embedding = history_embedding.to(device)       
        
        # edge weight embedding
        self.edge_embedding = nn.Embedding(50, e_dim)
        
        # time related embedding
        self.time2vec = Time2Vec(t2v_dim)
        self.parent_dia = Diachronic(parent_num, h_dim, gamma, activ)
        self.child_dia = Diachronic(child_num, h_dim, gamma, activ)
        
        # LSTM
        self.experience = nn.Linear(2 * h_dim + t2v_dim, h_dim)
        self.lstm1 = nn.LSTM(h_dim, h_dim, num_layers=rnn_layer, batch_first=True)  # get initial historical information
        self.lstm2 = nn.LSTM(h_dim, h_dim, num_layers=rnn_layer, batch_first=True)  # get sequence embedding
        
        # Graph Embedding Layer
        self.conv1 = HTGTLayer(h_dim, h_dim, e_dim, num_ntypes + 1, num_rels + 1, num_heads, device, t2v_activ=activ, mode="g2g", pre_norm=True)
        self.conv2 = HTGTLayer(h_dim, h_dim, e_dim, num_ntypes + 1, num_rels + 1, num_heads, device, t2v_activ=activ, mode="g2s", pre_norm=True)
        self.conv3 = HTGTLayer(h_dim, h_dim, e_dim, num_ntypes + 1, num_rels + 1, num_heads, device, t2v_activ=activ, mode="g2g", pre_norm=True)
        self.conv4 = HTGTLayer(h_dim, h_dim, e_dim, num_ntypes + 1, num_rels + 1, num_heads, device, t2v_activ=activ, mode="g2g", pre_norm=True)
        self.dropout = nn.Dropout(0.2)
        
        # link predictor for child nodes
        self.child_predictors = [
            nn.Sequential(nn.Linear(2 * h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, 1)) for _ in range(num_rels)
        ]
        for i, predictor in enumerate(self.child_predictors):
            for module in predictor:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=1.414)
            self.add_module('child_predictor_{}'.format(i), predictor)

        # link predictor for parent nodes
        self.parent_predictors = [
            nn.Sequential(nn.Linear(2 * h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, 1)) for _ in range(num_rels)
        ]
        for i, predictor in enumerate(self.parent_predictors):
            for module in predictor:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=1.414)
            self.add_module('parent_predictor_{}'.format(i), predictor)
        
        # attention scheme for merge individuals / childrens
        self.child_pool = HeteLinear(h_dim, h_dim, num_ntypes + 1)
        self.child_map = HeteLinear(h_dim, h_dim, num_ntypes + 1)
        self.child_attn = TypedLinear(h_dim, 1, num_ntypes + 1)
        self.indiv_pool = HeteLinear(h_dim, h_dim, num_ntypes + 1)
        self.indiv_attn = TypedLinear(h_dim, 1, num_ntypes + 1)
        
        # residual input
        self.skip = nn.Parameter(torch.ones(num_ntypes + 1))
        
        
    def __repr__(self):
        return "UniTRep"
    

    def forward(self, g_batch, s_batch):
        device = self.device
        
        # Seuquential Modeling
        seq_embed = self.sequence_module(s_batch, g_batch, device)     
        uniq_node, node_embed = self.individual_pooling(seq_embed, s_batch, device, pool=self.traj_pool)
        self.push_batch(uniq_node, node_embed)
        
        # Graph Modeling
        child_embed, parent_embed = self.graph_module(g_batch, device, pool=self.mj_pool)
        
        return seq_embed, child_embed, parent_embed


    def sequence_module(self, s_batch, g_batch, device):
        """ Embedding for each experience, with the consideration of effect from crowd flow

        Parameters
        ----------
        s_batch
            _description_
        g_batch
            _description_
        device
            _description_

        Returns 
        -------
            _description_
        """
        ########### Historical Information ###########
        length = s_batch['seq_len'].squeeze()
        seq = s_batch['sub_s'].long().to(device)
        dur = s_batch['seq_dur'].long().to(device)
        stime = s_batch['seq_stime'].long().to(device)
        etime = s_batch['seq_etime'].long().to(device)
        
        seq_embed = self.embedding(seq)
        stime_embed = self.child_dia(seq, seq_embed, stime.unsqueeze(-1))
        etime_embed = self.child_dia(seq, seq_embed, etime.unsqueeze(-1))
        seq_embed = torch.cat((stime_embed, etime_embed, self.time2vec(dur)), dim=-1)
        seq_embed = F.relu(self.experience(seq_embed))
        
        h0, c0 = self.init_hidden(seq.shape[0])
        packed_input = pack_padded_sequence(seq_embed, length, batch_first=True)
        packed_output, (_, _) = self.lstm1(packed_input, (h0, c0))
        seq_embed, _ = pad_packed_sequence(packed_output, batch_first=True)
        his_embed = packed_output.data              # historical embedding of sequence entity: [N, H]
        
        ########### Labor Market Influence ###########
        blocks = [block.to(device) for block in g_batch['sub_g']]

        sub_g = blocks[0]
        seed_nodes = blocks[0].srcdata[dgl.NID]
        x = F.relu(self.conv1(g=sub_g, 
                              src_h=self.embedding(seed_nodes), 
                              src_tw=self.child_dia.w[seed_nodes],
                              src_tb=self.child_dia.b[seed_nodes],
                              edge_h=self.edge_embedding(sub_g.edata['weight'])))
        x = self.dropout(x)
        
        # sort result embedding by dst ID
        dst_nodes = sub_g.dstdata[dgl.NID]
        x = x[torch.sort(dst_nodes)[1]]
        
        sub_g = g_batch['s_block'].to(device)           # g2s block
        graph_nodes = sub_g.srcdata[dgl.NID][len(sub_g.dstdata[dgl.NID]):]
        seq_nodes = pack_padded_sequence(seq.unsqueeze(-1), length, batch_first=True).data.squeeze()        # sequence entity ID: [N, ]
        _, indices = torch.unique(torch.cat((graph_nodes, seq_nodes, dst_nodes)), return_inverse=True)
        
        x = F.relu(self.conv2(g=sub_g, 
                              src_h=x[indices[:len(graph_nodes)]],
                              src_tw=self.child_dia.w[graph_nodes],
                              src_tb=self.child_dia.b[graph_nodes],
                              edge_h=self.edge_embedding(sub_g.edata['weight']),
                              dst_h=x[indices[len(graph_nodes) : len(graph_nodes) + len(seq_nodes)]],
                              dst_his=his_embed,
                              dst_tw=self.child_dia.w[seq_nodes],
                              dst_tb=self.child_dia.b[seq_nodes]))
        x = self.dropout(x)
        
        assert x.shape == his_embed.shape
        
        ########### Individual Trajectory ###########
        # update sequence embedding
        seq_embed = seq_embed.transpose(0, 1).reshape(-1, x.shape[-1])
        seq_embed[s_batch['valid_id']] = x
        seq_embed = seq_embed.reshape(seq.shape[1], seq.shape[0], x.shape[-1]).transpose(0, 1)
        packed_input = pack_padded_sequence(seq_embed, length, batch_first=True)
        
        assert torch.sum(packed_input.data - x) == 0
        
        packed_output, (_, _) = self.lstm2(packed_input, (h0, c0))
        seq_embed, _ = pad_packed_sequence(packed_output, batch_first=True)  # [B, L, H]
        
        return seq_embed
    
    
    def individual_pooling(self, seq_embed, s_batch, device, pool='attn'):
        """ Merge the embeddings of the same node in different sequences. (except for 0)
        todo: attention scheme differs the node types; refer to Heco
        
        Parameters
        ----------
        seq_embed
            _description_
        """
        seq = torch.LongTensor(s_batch['sub_s'])
        length = s_batch['seq_len'].squeeze()
        ntype = torch.LongTensor(s_batch['sub_ntype']).to(device)
        
        entity_embed = pack_padded_sequence(seq_embed, length, batch_first=True).data
        node_id = pack_padded_sequence(seq.unsqueeze(-1), length, batch_first=True).data.squeeze()
        ntype = pack_padded_sequence(ntype.unsqueeze(-1), length, batch_first=True).data.squeeze()
        
        uniq_node, indices = torch.unique(node_id, return_inverse=True)
        index = torch.cat((indices[None, :], torch.arange(entity_embed.shape[0])[None, :]), dim=0).to(device)
        if pool == "attn":
            attn = torch.tanh(self.indiv_pool(entity_embed, ntype))
            attn = self.indiv_attn(attn, ntype).flatten()
            weight = torch.sparse_coo_tensor(indices=index, values=attn, size=(len(uniq_node), entity_embed.shape[0]))
        elif pool == "avg":
            values = torch.ones(entity_embed.shape[0]).to(device)
            weight = torch.sparse_coo_tensor(indices=index, values=values, size=(len(uniq_node), entity_embed.shape[0]))
        else:
            raise NotImplementedError
        
        weight = torch.sparse.softmax(weight, dim=1)
        node_embed = torch.sparse.mm(weight, entity_embed)
        
        return uniq_node, node_embed
    
        
    def graph_module(self, g_batch, device, pool="avg"):
        """ Modeling the crowd flow graph with 2-layer HTGH

        Parameters
        ----------
        g_batch
            _description_
        device
            _description_

        Returns
        -------
            _description_
        """
        blocks = [block.to(device) for block in g_batch['sub_g']]
        c2p_map = g_batch['c2p_map'].to(device)
        
        sub_g = blocks[0]
        seed_nodes = sub_g.srcdata[dgl.NID]
        
        # Residual
        ntype = sub_g.srcdata['ntype']
        alpha = torch.sigmoid(self.skip[ntype]).unsqueeze(-1)
        x = self.embedding(seed_nodes) * alpha + self.pull(seed_nodes) * (1 - alpha)
        
        x = F.relu(self.conv3(g=sub_g, 
                              src_h=x, 
                              src_tw=self.child_dia.w[seed_nodes],
                              src_tb=self.child_dia.b[seed_nodes],
                              edge_h=self.edge_embedding(sub_g.edata['weight'])))
        x = self.dropout(x)
        
        sub_g = blocks[1]
        seed_nodes = sub_g.srcdata[dgl.NID]
        x = self.conv4(g=sub_g, 
                       src_h=x, 
                       src_tw=self.child_dia.w[seed_nodes],
                       src_tb=self.child_dia.b[seed_nodes],
                       edge_h=self.edge_embedding(sub_g.edata['weight']))
        child_embed = self.dropout(x)
        
        # Pooling to parent graph
        if pool == "attn":
            ntype = blocks[1].dstdata['ntype']
            trans_out = self.child_map(child_embed, ntype)
            attn = self.child_pool(child_embed, ntype)
            attn = self.child_attn(torch.tanh(attn), ntype).flatten()
            c2p_map = torch.sparse_coo_tensor(indices=c2p_map,
                                              values=attn,
                                              size=(c2p_map[0].max().item()+1,c2p_map[1].max().item()+1))
            c2p_map = torch.sparse.softmax(c2p_map, dim=1)
            parent_embed = torch.sparse.mm(c2p_map, child_embed)
        elif pool == "avg":
            c2p_map = torch.sparse_coo_tensor(indices=c2p_map,
                                              values=torch.ones(c2p_map.shape[1], device=device),
                                              size=(c2p_map[0].max().item()+1,c2p_map[1].max().item()+1))
            c2p_map = torch.sparse.softmax(c2p_map, dim=1)
            parent_embed = torch.sparse.mm(c2p_map, child_embed)
        else:
            raise NotImplementedError      
    
        return child_embed, parent_embed
    
    
    def init_hidden(self, batch_size):
        """ Initialize h0, c0
        # TODO: Refine this part with individual characters
        
        Parameters
        ----------
        batch_size
            _description_

        Returns
        -------
            _description_
        """
        h = torch.zeros((1 * self.rnn_layer, batch_size, self.h_dim), device=self.device)
        c = torch.zeros((1 * self.rnn_layer, batch_size, self.h_dim), device=self.device)
        return h, c
    

    def get_seq_loss(self, seq_embed, graph_embed, s_batch, k=500, tau=0.8, seq_loss="contra"):
        seq = torch.LongTensor(s_batch['sub_s'])
        h = seq_embed[:, :-1, :].reshape(-1, self.h_dim)
        uniq_node, indices = torch.unique(seq, return_inverse=True)
        
        # positive targets
        pos = indices[:, 1:].flatten()
        pos_mask = pos != 0
        pos_id = seq[:, 1:].flatten()   
        stime = torch.LongTensor(s_batch['seq_stime'])[:, 1:].flatten()
        
        pos_embed = graph_embed[pos][pos_mask]
        pos_id = pos_id[pos_mask]
        stime = stime[pos_mask].to(self.device)
        pos_embed = self.child_dia(pos_id, pos_embed, stime.unsqueeze(-1))  
        
        ###### Contrastive Loss ######
        h = h[pos_mask]
        pos_score = torch.exp(F.cosine_similarity(h, pos_embed) / tau)
        
        ntype = torch.LongTensor(s_batch['sub_ntype'])[:, 1:].flatten()
        neg_samples = negtive_sampler(pos[pos_mask], ntype[pos_mask], num_ntype=self.num_ntypes, neg_num=k)      # fixme: conflict
        neg_embed = graph_embed[neg_samples]
        neg_id = uniq_node[neg_samples]
        neg_embed = self.child_dia(neg_id, neg_embed, stime[:, None, None].repeat(1, k, 1))
        
        neg_score = F.cosine_similarity(h.unsqueeze(1).repeat(1, k, 1), neg_embed, dim=-1)
        neg_score = torch.exp(neg_score / tau).sum(1)
        
        if seq_loss == "contra":
            loss = -torch.log(torch.div(pos_score, pos_score + neg_score + 1e-8)).mean()
        elif seq_loss == "bpr":
            loss = -torch.log(torch.sigmoid(pos_score * k - neg_score)).mean()
        
        return loss
    

    def get_graph_loss(self, embed, nodes, edges, labels, neg_num=10, ntype='child'):
        """ Calculate the link prediction loss in HTG

        Parameters
        ----------
        embed
            node embedding
        nodes
            raw ID of nodes
        edges
            relabeled ID of edges
        labels
            _description_
        ntype, optional
            _description_, by default 'child'

        Returns
        -------
            _description_
        """
        src = edges[:, 0]
        dst = edges[:, 2]
        date = edges[:, 3].to(self.device)
        score = torch.zeros(len(edges), device=self.device)
        nodes = torch.LongTensor(nodes).to(self.device)
        
        for rel in range(self.num_rels):
            rel_mask = torch.where(edges[:, 1] == rel)
            if ntype == "child":
                src_embed = self.child_dia(nodes[src[rel_mask]], embed[src[rel_mask]], date[rel_mask].unsqueeze(-1))
                dst_embed = self.child_dia(nodes[dst[rel_mask]], embed[dst[rel_mask]], date[rel_mask].unsqueeze(-1))
                score[rel_mask] = self.child_predictors[rel](torch.cat((src_embed, dst_embed), dim=1)).flatten()
            else:
                src_embed = self.parent_dia(nodes[src[rel_mask]], embed[src[rel_mask]], date[rel_mask].unsqueeze(-1))
                dst_embed = self.parent_dia(nodes[dst[rel_mask]], embed[dst[rel_mask]], date[rel_mask].unsqueeze(-1))
                score[rel_mask] = self.parent_predictors[rel](torch.cat((src_embed, dst_embed), dim=1)).flatten()
                
        loss = F.binary_cross_entropy_with_logits(score, labels, pos_weight=torch.tensor([neg_num], device=self.device))

        return loss


    @torch.no_grad()
    def inference(self, g, c2p_map, g_bs, sequence=None, s_bs=None):
        """ Get Embedding for persons and entities

        Parameters
        ----------
        g
            crow flow graph
        sequence
            individual trajectories
        c2p_map
            hierarchical graph mapping
        g_bs
            graph batch size
        s_bs
            sequence batch size
        """
        device = self.device
        child_ntype = g.ndata['ntype'].to(device)
        
        ############# 1-hop Labor Market Influence #############
        # 1-hop neighbors
        g_dataloader = DataLoader(g,
                                  torch.arange(g.num_nodes()),
                                  graph_sampler=MultiLayerFullNeighborSampler(1),
                                  device=device,
                                  batch_size=g_bs,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=0)
        
        pre_embed = torch.zeros(g.num_nodes(), self.h_dim, device=device)       # node embedding incorporate 1-hop neighbor information
        for _, (input_nodes, output_nodes, blocks) in enumerate(g_dataloader):
            x = self.conv1(g=blocks[0].to(device), 
                           src_h=self.embedding(input_nodes), 
                           src_tw=self.child_dia.w[input_nodes],
                           src_tb=self.child_dia.b[input_nodes],
                           edge_h=self.edge_embedding(blocks[0].edata['weight']))
            pre_embed[output_nodes] = F.relu(x)
        
        # 2-hop neighbors
        row, all_h = [], []
        s_dataloader = torch.utils.data.DataLoader(SeqDataset(g, sequence, batch_size=s_bs), num_workers=4, prefetch_factor=2)
        for it, s_batch in enumerate(tqdm.tqdm(s_dataloader, desc="Sequence Inference")):
            s_batch = sbatch_convert(s_batch)
            
            ########### Historical Information ###########
            seq = s_batch['sub_s'].long().to(device)   # absolute ID
            block = s_batch['block'].to(device)
            dur = s_batch['dur'].long().to(device)
            stime = s_batch['stime'].long().to(device)
            etime = s_batch['etime'].long().to(device)
            seq_len = s_batch['seq_len'].long()
            
            seq_embed = self.embedding(seq)
            stime_embed = self.child_dia(seq, seq_embed, stime.unsqueeze(-1))
            etime_embed = self.child_dia(seq, seq_embed, etime.unsqueeze(-1))
            seq_embed = torch.cat((stime_embed, etime_embed, self.time2vec(dur)), dim=-1)
            seq_embed = F.relu(self.experience(seq_embed))

            h0, c0 = self.init_hidden(seq.shape[0])
            packed_input = pack_padded_sequence(seq_embed, seq_len, batch_first=True)
            packed_output, (_, _) = self.lstm1(packed_input, (h0, c0))
            seq_embed, _ = pad_packed_sequence(packed_output, batch_first=True)  # [B, L, H]
            his_embed = packed_output.data
            
            ########### Labor Market Influence ###########
            graph_nodes = block.srcdata[dgl.NID][block.num_dst_nodes():]
            seq_nodes = pack_padded_sequence(seq.unsqueeze(-1), seq_len, batch_first=True).data.squeeze()
            x = F.relu(self.conv2(g=block, 
                                  src_h=pre_embed[graph_nodes], 
                                  src_tw=self.child_dia.w[graph_nodes], 
                                  src_tb=self.child_dia.b[graph_nodes], 
                                  edge_h=self.edge_embedding(block.edata['weight']),
                                  dst_h=pre_embed[seq_nodes],
                                  dst_his=his_embed,
                                  dst_tw=self.child_dia.w[seq_nodes],
                                  dst_tb=self.child_dia.b[seq_nodes]))
            
            assert x.shape == his_embed.shape
            
            # update sequence embedding
            seq_embed = seq_embed.transpose(0, 1).reshape(-1, x.shape[-1])
            seq_embed[s_batch['valid_id']] = x
            seq_embed = seq_embed.reshape(seq.shape[1], seq.shape[0], x.shape[-1]).transpose(0, 1)
            packed_input = pack_padded_sequence(seq_embed, seq_len, batch_first=True)
            
            assert torch.sum(packed_input.data - x) == 0
            
            # final LSTM
            entity_embed = self.lstm2(packed_input, (h0, c0))[0].data
            
            row.append(seq_nodes)
            all_h.append(entity_embed)
            
            # if (it + 1) % 20 == 0:
            #     break
        
        # Pooling
        row = torch.cat(row)
        col = torch.arange(len(row)).to(device)
        all_h = torch.cat(all_h, dim=0)
        
        if self.traj_pool == "attn":
            ntype = child_ntype[row]
            attn = torch.tanh(self.indiv_pool(all_h, ntype))
            score = self.indiv_attn(attn, ntype).flatten()
        else:
            score = torch.ones(all_h.shape[0], device=device)
            
        traj_P = torch.sparse_coo_tensor(indices=torch.cat((row[None, :], col[None, :])),
                                         values=score, size=(g.num_nodes(), len(row)))
        traj_P = torch.sparse.softmax(traj_P, dim=1)
        feat = torch.sparse.mm(traj_P, all_h)
            
        ########## Crowd Flow Modeling ##########
        g_dataloader = DataLoader(g,
                                  torch.arange(g.num_nodes()),
                                  graph_sampler=MultiLayerFullNeighborSampler(2),
                                  device=device,
                                  batch_size=1000,          # note: 训练数据太大了图比较稠密，调低一点
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=0)
        
        child_embed = torch.zeros(g.num_nodes(), self.h_dim).to(device)
        for _, (input_nodes, output_nodes, blocks) in enumerate(g_dataloader):
            sub_g = blocks[0].to(device)
            ntype = sub_g.srcdata['ntype']
            alpha = torch.sigmoid(self.skip[ntype]).unsqueeze(-1)
            x = self.embedding(input_nodes) * alpha + feat[input_nodes] * (1 - alpha)
            
            h = F.relu(self.conv3(g=sub_g, 
                                  src_h=x, 
                                  src_tw=self.child_dia.w[input_nodes],
                                  src_tb=self.child_dia.b[input_nodes],
                                  edge_h=self.edge_embedding(sub_g.edata['weight'])))
            
            sub_g = blocks[1].to(device)
            seed_nodes = sub_g.srcdata[dgl.NID]
            h = self.conv4(g=sub_g, 
                           src_h=h, 
                           src_tw=self.child_dia.w[seed_nodes],
                           src_tb=self.child_dia.b[seed_nodes],
                           edge_h=self.edge_embedding(sub_g.edata['weight']))
            child_embed[output_nodes] = h
        
        ########## Pooling from M&J Graph => S&C Graph ##########
        child_ntype = g.ndata['ntype'].to(device)
        indices = c2p_map.coalesce().indices().to(device)       # [parent, child]
        
        if self.mj_pool == "attn":
            ntype = child_ntype[indices[1]]
            trans_out = self.child_map(child_embed[indices[1]], ntype)
            attn = self.child_pool(child_embed[indices[1]], ntype)
            attn = self.child_attn(torch.tanh(attn), ntype).flatten()
            indiv_P = torch.sparse_coo_tensor(indices=indices,
                                              values=attn,
                                              size=c2p_map.coalesce().size())
            indiv_P = torch.sparse.softmax(indiv_P, dim=1)
            # parent_embed = torch.sparse.mm(indiv_P, trans_out)
            parent_embed = torch.sparse.mm(indiv_P, child_embed)
        elif self.mj_pool == "avg":
            indiv_P = torch.sparse.softmax(c2p_map.to(device), dim=1)
            parent_embed = torch.sparse.mm(indiv_P, child_embed)
        else:
            raise NotImplementedError
        
        return child_embed, parent_embed, pre_embed
    

    @torch.no_grad()
    def push_batch(self, nids, x):
        """ 更新一个 batch 的节点 """
        self.history_embedding[nids] = x.clone().detach()


    @torch.no_grad()
    def pull(self, nids):
        res = self.history_embedding[nids]
        return res
    


import math
import torch
import torch.nn as nn

import dgl
import dgl.function as fn
from dgl.nn import TypedLinear
from dgl.nn.functional import edge_softmax

class HTGTLayer(nn.Module):
    """
    Heterogenous Temporal Graph Transformer Layer
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 e_dim,
                 num_ntypes,
                 num_rels,
                 num_heads,
                 device,
                 dropout=0.2,
                 bias=True,
                 layer_norm=True,
                 pre_norm=False,
                 self_loop=True,
                 t2v_activ='sin',
                 mode='g2g'):
        
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_ntypes = num_ntypes
        self.num_rels = num_rels
        self.device = device
        self.self_loop = self_loop
        self.layer_norm = layer_norm
        self.pre_norm = pre_norm
        self.bias = bias
        self.mode = mode
        self.num_heads = num_heads
        self.head_size = out_dim // num_heads
        
        # self-attention
        self.linear_k = TypedLinear(in_dim + e_dim, out_dim, num_rels)
        self.linear_v = TypedLinear(in_dim + e_dim, out_dim, num_rels)
        self.linear_a = TypedLinear(out_dim, out_dim, num_ntypes)
        if mode == 'g2s':
            self.linear_q = TypedLinear(2 * in_dim, out_dim, num_rels)
        else:
            self.linear_q = TypedLinear(in_dim, out_dim, num_rels)
        
        # tim2vec activation
        if t2v_activ == "sin":
            self.f = torch.sin
        else:
            self.f = torch.cos
        
        # bias
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(num_ntypes, out_dim))
            nn.init.zeros_(self.h_bias)
        
        # norm
        if self.layer_norm:
            self.src_norm = nn.LayerNorm(in_dim + e_dim)
            if mode == 'g2g': 
                self.dst_norm = nn.LayerNorm(in_dim)
            else:
                self.dst_norm = nn.LayerNorm(in_dim * 2)
        
        self.skip = nn.Parameter(torch.ones(num_ntypes))
        self.drop = nn.Dropout(dropout)
        
    def message(self, edges):
        """ Message Passing with DotProductAttention"""
        t = edges.data['date'].view(-1, 1)
        time_dim = edges.src['t_w'].shape[-1]
        
        dia_src = self.f(edges.src['t_w'] * t + edges.src['t_b'])
        dia_dst = self.f(edges.dst['t_w'] * t + edges.dst['t_b'])
        dia_src = torch.cat((dia_src * edges.src['h'][:, :time_dim], edges.src['h'][:, time_dim:]), dim=-1)
        dia_dst = torch.cat((dia_dst * edges.dst['h'][:, :time_dim], edges.dst['h'][:, time_dim:]), dim=-1)
        
        weight_embed = edges.data['h']
        dia_src = torch.cat((dia_src, weight_embed), dim=1)     # [E, H+H']
        if self.mode == 'g2s':
            dia_dst = torch.cat((dia_dst, edges.dst['his']), dim=-1)    # [E, 2*H]
        
        if self.pre_norm:
            dia_src = self.src_norm(dia_src)
            dia_dst = self.dst_norm(dia_dst)

        q = self.linear_q(dia_dst, edges.data[dgl.ETYPE]).view(-1, self.num_heads, self.head_size)
        k = self.linear_k(dia_src, edges.data[dgl.ETYPE]).view(-1, self.num_heads, self.head_size)
        v = self.linear_v(dia_src, edges.data[dgl.ETYPE]).view(-1, self.num_heads, self.head_size)   # [E, num_heads, H]
        
        attn_score = (q * k).sum(-1) / math.sqrt(self.out_dim)      # [E, num_heads]
        
        return {'a': attn_score, 'm': v}
    
    def forward(self, g, src_h, src_tw, src_tb, edge_h, dst_h=None, dst_his=None, dst_tw=None, dst_tb=None):
        with g.local_scope():
            if self.mode == 'g2g':
                g.srcdata['h'] = src_h
                g.srcdata['t_w'] = src_tw
                g.srcdata['t_b'] = src_tb
                dst_h = src_h[:g.num_dst_nodes()]
                g.dstdata['h'] = dst_h
                g.dstdata['t_w'] = src_tw[:g.num_dst_nodes()]
                g.dstdata['t_b'] = src_tb[:g.num_dst_nodes()]
                g.edata['h'] = edge_h
            else:
                g.srcdata['h'] = torch.cat((dst_his, src_h), dim=0)     # sequence_embedding || graph_embedding
                g.srcdata['t_w'] = torch.cat((dst_tw, src_tw), dim=0)
                g.srcdata['t_b'] = torch.cat((dst_tb, src_tb), dim=0)
                g.dstdata['h'] = dst_h
                g.dstdata['his'] = dst_his
                g.dstdata['t_w'] = dst_tw
                g.dstdata['t_b'] = dst_tb
                g.edata['h'] = edge_h
            
            # Post-LN: MultiHeadAttn + Residual + LayerNorm
            # Pre-LN: LayerNorm + MultiHeadAttn + Residual
            
            # message passing
            g.apply_edges(self.message)
            g.edata['m'] = g.edata['m'] * edge_softmax(g, g.edata['a']).unsqueeze(-1)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h'))
            h = g.dstdata['h'].view(-1, self.out_dim)
            ntype = g.dstdata['ntype']
            
            # apply bias
            if self.bias:
                h = h + self.h_bias[ntype]
                
            # target-specific residual connection
            if self.self_loop:
                alpha = torch.sigmoid(self.skip[ntype]).unsqueeze(-1)
                h = self.drop(self.linear_a(h, ntype))
                h = h * alpha + dst_h * (1 - alpha)
            
            # LayerNorm
            if not self.pre_norm and self.layer_norm:
                h = self.norm(h)
            
            return h
        
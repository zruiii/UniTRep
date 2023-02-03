
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from dgl.nn import TypedLinear

class Time2Vec(nn.Module):

    def __init__(self, h_dim, activ='sin'):
        super(Time2Vec, self).__init__()
        self.w0 = nn.Parameter(torch.rand(1, 1))
        self.b0 = nn.Parameter(torch.rand(1, 1))
        self.w = nn.Parameter(torch.rand(1, h_dim - 1))
        self.b = nn.Parameter(torch.rand(1, h_dim - 1))
        
        nn.init.normal_(self.w)
        nn.init.normal_(self.b)

        if activ == "sin":
            self.f = torch.sin
        else:
            self.f = torch.cos

    def forward(self, scalar):
        scalar = scalar.unsqueeze(-1).to(torch.float32)  # [N, 1]
        v0 = torch.matmul(scalar, self.w0) + self.b0  # [N, 1]
        v1 = self.f(torch.matmul(scalar, self.w) + self.b)  # [N, H-1]
        return torch.cat((v0, v1), dim=-1)  # [N, H]


class Diachronic(nn.Module):
    """ Diachronic Embedding for each node.
    Hold the assumption that the importance of a partion of dimensions are time-variant.
    When the gamma == 0, they donnot change along the time.
    """
    
    def __init__(self, num_nodes, h_dim, gamma, activ='sin'):
        super(Diachronic, self).__init__()
        self.t_dim = int(h_dim * gamma)
        self.w = nn.Parameter(torch.randn(num_nodes, self.t_dim))
        self.b = nn.Parameter(torch.randn(num_nodes, self.t_dim))
        
        nn.init.normal_(self.w)
        nn.init.normal_(self.b)
        
        if activ == 'sin':
            self.f = torch.sin
        else:
            self.f = torch.cos
    
    def forward(self, nid, x, t):
        """ Return the embeddings of nid nodes at time t.
        
        Parameters
        ----------
        nid
            absolute node IDs
        x
            node embeddings [N, H] or [B, L, H]
        t
            timestamps [N, 1] or [B, L, 1]

        Returns
        -------
            diachronic embeddings
        """
        time_embed = self.f(self.w[nid] * t + self.b[nid])
        x = torch.cat((time_embed * x[..., :self.t_dim], x[..., self.t_dim:]), dim=-1)
        return x


class HeteLinear(nn.Module):
    """ Expand TypedLinear with bias """
    def __init__(self, in_size, out_size, num_types):
        super(HeteLinear, self).__init__()
        
        self.linear = TypedLinear(in_size, out_size, num_types)
        self.bias = Parameter(torch.empty(num_types, out_size))
        self.reset_bias()
        
    def forward(self, x, x_type):
        x = self.linear(x, x_type)
        x = x + self.bias[x_type]
        return x
    
    def reset_bias(self):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.get_weight())
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound) 


       
if __name__ == "__main__":
    model = HeteLinear(8, 8, 2)
    x = torch.randn((6, 8))
    ntype = torch.tensor([0, 1, 1, 0, 1, 0])
    print(model(x, ntype))

    
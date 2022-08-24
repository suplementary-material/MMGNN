import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class MMConv(nn.Module):
    def __init__(self, in_features, out_features,  moment=3, use_center_moment=False):
        super(MMConv, self).__init__() 
        self.moment = moment
        self.use_center_moment = use_center_moment
        self.in_features = in_features

        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.w_att = Parameter(torch.FloatTensor(self.in_features * 2,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.w_att.data.uniform_(-stdv, stdv)
    def moment_calculation(self, x, adj_t, moment):
        mu = torch.spmm(adj_t, x)
        out_list = [mu]
        if moment > 1:
            if self.use_center_moment:
                sigma = torch.spmm(adj_t, (x - mu).pow(2))
            else:
                sigma = torch.spmm(adj_t, (x).pow(2))
            sigma[sigma == 0] = 1e-16
            sigma = sigma.sqrt()
            out_list.append(sigma)

            for order in range(3, moment+1):
                gamma = torch.spmm(adj_t, x.pow(order))
                mask_neg = None
                if torch.any(gamma == 0):
                    gamma[gamma == 0] = 1e-16
                if torch.any(gamma < 0):
                    mask_neg = gamma < 0
                    gamma[mask_neg] *= -1
                gamma = gamma.pow(1/order)
                if mask_neg != None:
                    gamma[mask_neg] *= -1
                out_list.append(gamma)
        return out_list
    def attention_layer(self, moments, q):
            k_list = []
            # if self.use_norm:
            #     h_self = self.norm(h_self) # ln
            q = q.repeat(self.moment, 1) # N * m, D
            # output for each moment of 1st-neighbors
            k_list = moments
            attn_input = torch.cat([torch.cat(k_list, dim=0), q], dim=1)
            attn_input = F.dropout(attn_input, 0.5, training=self.training)
            e = F.elu(torch.mm(attn_input, self.w_att)) # N*m, D
            attention = F.softmax(e.view(len(k_list), -1, self.out_features).transpose(0, 1), dim=1) # N, m, D
            out = torch.stack(k_list, dim=1).mul(attention).sum(1) # N, D
            return out
    def forward(self, input, adj , h0 , lamda, alpha, l, beta = 0.1):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        support = (1-alpha)*hi+alpha*h0
        r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        out_moment = self.attention_layer(self.moment_calculation(h0, adj, self.moment), output)
        output = (1 - beta) * output + beta * out_moment
        return output


class MMGNN(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, use_center_moment=False, moment=3):
        super(MMGNN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(MMConv(nhidden, nhidden, use_center_moment=use_center_moment, moment=moment))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)

if __name__ == '__main__':
    pass







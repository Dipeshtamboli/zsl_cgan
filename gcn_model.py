import pdb
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
# from pygcn.layers import GraphConvolution

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    split_1 = sio.loadmat("/home/SharedData/fabio/cgan/hmdb_i3d/split_1/att_splits.mat")
    att = split_1["att"]
    att = torch.tensor(att).cuda()
    att = torch.transpose(att,1,0)

    att = att[:51].cuda() #first 25 classes    
    att = att.type(torch.FloatTensor).cuda()
    model = GCN(nfeat=300,
                nhid=128,
                nclass=51,
                dropout=0.5).cuda()
    print(model)
    num_classes = 51
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    adj = torch.zeros((num_classes,num_classes)).cuda()
    for i in range(num_classes):
        for j in range(num_classes):
            adj[i][j] = cos(att[i].unsqueeze(0),att[j].unsqueeze(0))
    output = model(att, adj)
    pdb.set_trace()
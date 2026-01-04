#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """
    def __init__(self, in_features, out_features, bias=True, node_n=24):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ResBlock(nn.Module):
    """
    Define a residual block of GCN
    """
    def __init__(self, in_features, node_n=24, act_f=nn.Tanh(), p_dropout=0, is_bn=False):
        super(ResBlock, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.is_bn = is_bn

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=not is_bn)
        if is_bn:
            self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=not is_bn)
        if is_bn:
            self.bn2 = nn.BatchNorm1d(node_n * in_features)

        # self.do = nn.Dropout(p_dropout)
        self.act_f = act_f

    def forward(self, x):
        y = self.gc1(x)
        if self.is_bn:
            B, J, F = y.shape
            y = self.bn1(y.view(B, -1)).view(B, J, F)
        y = self.act_f(y)
        # y = self.do(y)

        y = self.gc2(y)
        if self.is_bn:
            B, J, F = y.shape
            y = self.bn2(y.view(B, -1)).view(B, J, F)
        y = self.act_f(y)
        # y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ResGCN(nn.Module):
    """
    Define a structure of residual GCN
    """
    def __init__(self, input_feature, hidden_feature, p_dropout=0, num_block=1, node_n=24, is_bn=False,
                 act_f=nn.Tanh()):
        super(ResGCN, self).__init__()
        self.num_block = num_block
        self.is_bn = is_bn

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        if is_bn:
            self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        gcbs = []
        for i in range(num_block):
            gcbs.append(ResBlock(hidden_feature, node_n=node_n, is_bn=is_bn, act_f=act_f))
        self.gcbs = nn.Sequential(*gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        # self.do = nn.Dropout(p_dropout)
        self.act_f = act_f

    def forward(self, x):
        y = self.gc1(x)
        if self.is_bn:
            B, J, F = y.shape
            y = self.bn1(y.view(B, -1)).view(B, J, F)
        y = self.act_f(y)
        # y = self.do(y)
        y = self.gcbs(y)
        y = self.gc7(y)
        y = y + x
        return y


class GCNParts(nn.Module):
    def __init__(self, input_feature, hidden_feature,
                 p_dropout=0, num_block=1, node_n=24, is_bn=False,
                 act_f=nn.Tanh()):
        """
        Args:
            input_feature (int): Dimension of input features per node.
            hidden_feature (int): Dimension of hidden features per node.
            p_dropout (float): Dropout probability.
            num_block (int): Number of residual GC blocks.
            node_n (int): Number of nodes (joints) in the graph.
            is_bn (bool): Whether to use batch normalization.
            act_f (nn.Module): Activation function.
        """
        super(GCNParts, self).__init__()

        gcns = []
        gcns.append(ResGCN(input_feature, hidden_feature, num_block=num_block,
                        node_n=node_n, is_bn=is_bn, act_f=act_f))
        self.gcns = nn.ModuleList(gcns)


    def forward(self, dct_coeff):
        """
        Args:
            dct_coeff (Tensor): Input DCT coefficient matrix
                shape: (B, J, 3M),
                where B is batch size, J is the number of joints,
                and M is the number of DCT bases.

        Returns:
            joint_feat (Tensor): Joint features
                shape: (B, J, F),
                where F is feature dimension of GCN output.
        """
        gcn_input  = dct_coeff.clone()
        joint_feat = self.gcns[0](gcn_input)

        return joint_feat

# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/11/27 14:14
# Project Name: MLCourse-IMDB
# File        : functional.py
# --------------------------------------------------

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import pdb


def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))


def reset_normal_param(L, stdv, weight_scale=1.):
    assert type(L) == torch.nn.Linear
    torch.nn.init.normal(L.weight, std=weight_scale / math.sqrt(L.weight.size()[0]))


class LinearWeightNorm(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_scale=None, weight_init_stdv=0.1):
        super(LinearWeightNorm, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.randn(out_features, in_features) * weight_init_stdv)
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        if weight_scale is not None:
            assert type(weight_scale) == int
            self.weight_scale = Parameter(torch.ones(out_features, 1) * weight_scale)
        else:
            self.weight_scale = 1

    def forward(self, x):
        W = self.weight * self.weight_scale / torch.sqrt(torch.sum(self.weight ** 2, dim=1, keepdim=True))
        return F.linear(x, W, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', weight_scale=' + str(self.weight_scale) + ')'


class KMaxPool1d(torch.nn.Module):
    def __init__(self, k: int):
        super(KMaxPool1d, self).__init__()
        self.k = k

    def forward(self, x):
        dim = len(x.size()) - 1
        index = x.topk(self.k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'k=' + str(self.k) \
               + ')'


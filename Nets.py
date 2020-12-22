# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/11/27 14:14
# Project Name: MLCourse-IMDB
# File        : Nets.py
# --------------------------------------------------

import torch
from torch.nn.parameter import Parameter
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import pdb
from functional import reset_normal_param, LinearWeightNorm, KMaxPool1d
from PreProcess import SENTENCE_LEN, WORD_DIM, SEED


MIDDLE_DIM = 80
NEG_SLOPE = 0.02
RNN_LAYERS = 2
KER_REPEAT = 2
KER_SIZES = [3, 4, 5, 6, 7] * KER_REPEAT
TOPK_NUM = 8


class Discriminator(nn.Module):
    def __init__(self, input_dim=SENTENCE_LEN * WORD_DIM, output_dim=2):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=WORD_DIM, out_channels=MIDDLE_DIM, kernel_size=ker),
                          nn.BatchNorm1d(num_features=MIDDLE_DIM),
                          nn.LeakyReLU(negative_slope=NEG_SLOPE),
                          # nn.MaxPool1d(kernel_size=SENTENCE_LEN - ker + 1),
                          KMaxPool1d(k=TOPK_NUM)
                          )
            for ker in KER_SIZES
        ])
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(KER_SIZES) * MIDDLE_DIM * TOPK_NUM, MIDDLE_DIM),
                nn.BatchNorm1d(num_features=MIDDLE_DIM),
                nn.LeakyReLU(negative_slope=NEG_SLOPE)
            ),
            nn.Sequential(
                nn.Linear(MIDDLE_DIM, output_dim)
            )
        ])

    def forward(self, x: torch.Tensor, feature=False, cuda=True):
        x = x.squeeze(dim=1).permute(dims=[0, 2, 1])
        after_convs = [conv(x) for conv in self.convs]
        after_convs = torch.cat(after_convs, dim=1)
        x = after_convs.view(after_convs.size()[0], -1)
        feature_matching = x
        noise = torch.randn(x.size()) * 0.05 if self.training else torch.Tensor([0])
        if cuda:
            noise = noise.cuda()
        x = x + Variable(noise, requires_grad=False)
        x_f = None
        for i in range(len(self.layers)):
            m = self.layers[i]
            x_f = m(x)
            noise = torch.randn(x_f.size()) * 0.05 if self.training else torch.Tensor([0])
            if cuda:
                noise = noise.cuda()
            x = (x_f + Variable(noise, requires_grad=False))
        if feature:
            return feature_matching, x_f
        return x_f


class Generator(nn.Module):
    def __init__(self, z_dim, output_dim=WORD_DIM):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(input_size=z_dim, hidden_size=output_dim, num_layers=RNN_LAYERS, batch_first=True)

    def forward(self, batch_size, cuda=True):
        z = Variable(torch.rand(batch_size, 1, self.z_dim), requires_grad=False)
        if cuda:
            z = z.cuda()
        h0 = torch.zeros(RNN_LAYERS, batch_size, self.output_dim)
        c0 = torch.zeros(RNN_LAYERS, batch_size, self.output_dim)
        if cuda:
            h0, c0 = h0.cuda(), c0.cuda()
        hn, cn = h0, c0
        x = z
        sentence = []
        for i in range(SENTENCE_LEN):
            x, (hn, cn) = self.rnn(x, (hn, cn))
            sentence.append(x.squeeze(dim=1))
        return torch.stack(sentence, dim=1).unsqueeze(dim=1)

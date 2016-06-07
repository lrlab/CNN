# -*- coding: utf-8 -*-

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

"""
Code for Convolutional Neural Networks for Sentence Classification

author: ichiroex
"""

class CNNSC(ChainList):
    def __init__(self,
                 input_channel,
                 output_channel,
                 filter_height,
                 filter_width,
                 n_label,
                 max_sentence_len):
        link_list = []
        link_list += [L.Convolution2D(input_channel, output_channel, (i, filter_width), pad=0) for i in filter_height]
        link_list += [L.Linear(output_channel * 3, output_channel * 3), L.Linear(output_channel * 3, n_label)]

        super(CNNSC, self).__init__(*link_list)

        self.max_sentence_len = max_sentence_len
        self.filter_height = filter_height
        self.cnv_num = len(filter_height)

    def __call__(self, x, train=True):
        
        h_conv = [None for i in self.filter_height]
        h_pool = [None for i in self.filter_height]
        
        for i in self.filter_height:
            h_conv[i] = F.relu(self[i](x))
            h_pool[i] = F.max_pooling_2d(h_conv[i], self.max_sentence_len)
        concat = F.concat(h_pool, axis=2)
        
        
        h_l1 = F.dropout(F.tanh(self[self.cnv_num+0](concat)), ratio=0.5, train=train)
        y = self[self.cnv_num+1](h_l1)

        return y

if __name__ == '__main__':
    model = L.Classifier(CNNSC(input_channel=1,
                           output_channel=100,
                           filter_height=[3,4,5],
                           filter_width=20,
                           n_units=100,
                           n_label=2,
                           max_sentence_len=20))
    print('done process')

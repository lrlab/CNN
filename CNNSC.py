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
    
    max_sentence_len = 0

    def __init__(self,
                 input_channel,
                 output_channel,
                 filter_height,
                 filter_width,
#                  n_units,
                 n_label,
                 max_sentence_len):

        link_list = []
        link_list += [L.Convolution2D(input_channel, output_channel, (i, filter_width), pad=0) for i in filter_height]
        link_list += [L.Linear(output_channel * 3, output_channel * 3), L.Linear(output_channel * 3, n_label)]

        super(CNNSC, self).__init__(*link_list)
#             conv1=L.Convolution2D(input_channel, output_channel, (3, filter_width), pad=0),
#             conv2=L.Convolution2D(input_channel, output_channel, (4, filter_width), pad=0),
#             conv3=L.Convolution2D(input_channel, output_channel, (5, filter_width), pad=0),
#             l1=L.Linear(output_channel, n_units),
#             l2=L.Linear(output_channel, n_units),
#             l3=L.Linear(output_channel, n_units),
#             l4=L.Linear(n_units, n_label)
#         )

        self.max_sentence_len = max_sentence_len
        self.filter_height = filter_height
        self.cnv_num = len(filter_height)

#     def __call__(self, x):
#         h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 3)
#         h2 = F.dropout(F.relu(self.l1(h1)))
#         y = self.l2(h2)
#         return y


    def __call__(self, x, train=True):
        
        h_conv = [None for i in self.filter_height]
        h_pool = [None for i in self.filter_height]
        
        for i in self.filter_height:
            h_conv[i] = F.relu(self[i](x))
            h_pool[i] = F.max_pooling_2d(h_conv[i], self.max_sentence_len)
        concat = F.concat(h_pool, axis=2)
        
#         h_conv1 = F.relu(self[0](x))
#         h_conv2 = F.relu(self[1](x))
#         h_conv3 = F.relu(self[2](x))
        """
        print 'x:', x.data.shape
        print 'h_conv1:', h_conv1.data.shape
        print 'h_conv2:', h_conv2.data.shape
        print 'h_conv3:', h_conv3.data.shape
        #"""
#         h_pool1 = F.max_pooling_2d(h_conv1, self.max_sentence_len )
#         h_pool2 = F.max_pooling_2d(h_conv2, self.max_sentence_len )
#         h_pool3 = F.max_pooling_2d(h_conv3, self.max_sentence_len )
#         concat = F.concat((h_pool1, h_pool2, h_pool3), axis=2)
        """
        print 'h_pool1:', h_pool1.data.shape
        print 'h_pool2:', h_pool2.data.shape
        print 'h_pool3:', h_pool3.data.shape
        print "concat.data.shape:", concat.data.shape
        #"""

        h_l1 = F.dropout(F.tanh(self[self.cnv_num+0](concat)), ratio=0.5, train=train)
        y = self[self.cnv_num+1](h_l1)

        """
        print 'h_l1:', h_l1.data.shape
        print 'y:', y.data.shape
        print "==================="
        """
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

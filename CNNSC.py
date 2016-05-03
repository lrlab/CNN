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

class CNNSC(Chain):
    
    max_sentence_len = 0

    def __init__(self, input_channel, output_channel, filter_height, filter_width, n_units, n_label, max_sentence_len):

        super(CNNSC, self).__init__(
            conv1=L.Convolution2D(input_channel, output_channel, (3, filter_width), pad=0),
            conv2=L.Convolution2D(input_channel, output_channel, (4, filter_width), pad=0),
            conv3=L.Convolution2D(input_channel, output_channel, (5, filter_width), pad=0),
            l1=L.Linear(output_channel, n_units),
            l2=L.Linear(output_channel, n_units),
            l3=L.Linear(output_channel, n_units),
            l4=L.Linear(n_units, n_label)
        )

        self.max_sentence_len = max_sentence_len

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 3)
        h2 = F.dropout(F.relu(self.l1(h1)))
        y = self.l2(h2)
        return y


    def __call__(self, x, train=True):
        
        h_conv1 = F.relu(self.conv1(x))
        h_conv2 = F.relu(self.conv2(x))
        h_conv3 = F.relu(self.conv3(x))

        """
        print 'x:', x.data.shape
        print 'h_conv1:', h_conv1.data.shape
        print 'h_conv2:', h_conv2.data.shape
        print 'h_conv3:', h_conv3.data.shape
        #"""

        h_pool1 = F.max_pooling_2d(h_conv1, self.max_sentence_len )
        h_pool2 = F.max_pooling_2d(h_conv2, self.max_sentence_len )
        h_pool3 = F.max_pooling_2d(h_conv3, self.max_sentence_len )

        """
        print 'h_pool1:', h_pool1.data.shape
        print 'h_pool2:', h_pool2.data.shape
        print 'h_pool3:', h_pool3.data.shape
        #"""

        h_l1 = F.relu(self.l1(h_pool1))
        h_l2 = F.relu(self.l2(h_pool2))
        h_l3 = F.relu(self.l3(h_pool3))

        #h_l1_drop = F.dropout(F.concat((h_l1, h_l2, h_l3)), ratio=0.5, train=train)
        h_l1_drop = F.dropout(h_l1 + h_l2 + h_l3, ratio=0.5, train=train)
        y = self.l4(h_l1_drop)

        """
        print 'h_l1:', h_l1.data.shape
        print 'h_l2:', h_l2.data.shape
        print 'h_l3:', h_l3.data.shape
        print 'h_l1_drop:', h_l1_drop.data.shape
        print 'y:', y.data.shape
        #"""

        return y

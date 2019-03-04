# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:58:37 2019

@author: Administrator
"""

import tensorflow as tf

input_mode=784
output_mode=10
layers_mode=500

def get_weight(shape,regularizer):
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer !=None:tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w
    
def get_baise(shape):
    b=tf.Variable(tf.zeros(shape))
    return b

def farward(x,regularizer):
    w1=get_weight([input_mode,layers_mode],regularizer)
    b1=get_baise([layers_mode])
    y1=tf.nn.relu(tf.matmul(x,w1)+b1)
    
    w2=get_weight([layers_mode,output_mode],regularizer)
    b2=get_baise([output_mode])
    pd=tf.matmul(y1,w2)+b2

    return pd

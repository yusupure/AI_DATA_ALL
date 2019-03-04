# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 22:03:15 2019

@author: Administrator
"""

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import mnist_farward_00
import os

batch_size=200
lrb=0.1
lrd=0.99
moving_reta=0.99
model_save_path='../sc'
model_filename='mnist_model'
regularizer=0.00001

def backward(mnist):
    x=tf.placeholder(tf.float32,[None,mnist_farward_00.input_mode])
    y=tf.placeholder(tf.float32,[None,mnist_farward_00.output_mode])
    pd=mnist_farward_00.farward(x,regularizer)
    global_step=tf.Variable(0,trainable=False)
    #交叉商正则化
    cem=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pd,labels=tf.argmax(y,1)))
    loss=cem+tf.add_n(tf.get_collection('losses'))
    #自动化损失率
    last_step=tf.train.exponential_decay(lrb,global_step,mnist.train.num_examples/batch_size,lrd,staircase=True)
    train_step=tf.train.GradientDescentOptimizer(last_step).minimize(loss,global_step)
    #滑动平均
    ema=tf.train.ExponentialMovingAverage(moving_reta,global_step)
    ema_op=ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op(name='train')
    saver=tf.train.Saver()
    
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())
        tf.reset_default_graph()
        ckpt=tf.train.get_checkpoint_state(model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        for i in range(50000):
            xs,ys=mnist.train.next_batch(batch_size)
            _,loss_values,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y:ys})
            if i%1000==0:
                print(step,loss_values)
                saver.save(sess,os.path.join(model_save_path,model_filename),global_step=global_step)

def main():
    mnist=read_data_sets('mnist_data',one_hot=True)
    backward(mnist)
if __name__=='__main__':
    main()
        
        
        
        

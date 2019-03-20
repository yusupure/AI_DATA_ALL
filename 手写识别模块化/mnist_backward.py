# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:24:35 2019

@author: Administrator
"""

import tensorflow as tf
import os
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import mnist_farward
import mnist_tfrecord
batch_size=200
lrb=0.1
lrd=0.99
moving_rate=0.99
regularizer=0.00001
model_save_path='/sc'
model_file_name='mnist_model'
num_example_list=60000


def backward():
    x=tf.placeholder(tf.float32,[None,mnist_farward.input_mode])
    y=tf.placeholder(tf.float32,[None,mnist_farward.output_mode])
    pd=mnist_farward.farward(x,regularizer)
    global_step=tf.Variable(0,trainable=False)
    #交叉熵，正则化
    cem=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pd,labels=tf.argmax(y,1)))
    loss=cem+tf.add_n(tf.get_collection('losses'))
    #自动化学习率
    last_step=tf.train.exponential_decay(lrb,num_example_list/batch_size,global_step,lrd,staircase=True)
    train_step=tf.train.GradientDescentOptimizer(last_step).minimize(loss,global_step=global_step)
    #滑动平均
    ema=tf.train.ExponentialMovingAverage(moving_rate,global_step)
    ema_op=ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op(name='train')
    saver=tf.train.Saver()
    #读取自建数据集处理方法
    img_batch,labels_batch=mnist_tfrecord.Tfrecorde_load(batch_size,isTrain=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.reset_default_graph()
        ckpt=tf.train.get_checkpoint_state(model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        
        coord=tf.train.Coordinator()
        treades=tf.train.start_queue_runners(sess=sess,coord=coord)
        for i in range(50000):
            #自动数据集处理应用数据
            xs,ys=sess.run([img_batch,labels_batch])
            #xs,ys=mnist.train.next_batch(batch_size)
            _,loss_values,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y:ys})
            if i%1000==0:
                print(step,loss_values)
                saver.save(sess,os.path.join(model_save_path,model_file_name),global_step=global_step)
        
        coord.clear_stop()
        coord.join(treades)
                
def main():
    #mnist=read_data_sets('mnist_data',one_hot=True)
    #backward(mnist)
    backward()
if __name__=='__main__':
    main()

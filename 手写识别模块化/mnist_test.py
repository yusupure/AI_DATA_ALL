# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 22:20:31 2019

@author: Administrator
"""

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import mnist_backward_00
import mnist_farward_00
import time

def test(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[None,mnist_farward_00.input_mode])
        y=tf.placeholder(tf.float32,[None,mnist_farward_00.output_mode])
        pd=mnist_farward_00.farward(x,None)
        
        #滑动平均
        ema=tf.train.ExponentialMovingAverage(mnist_backward_00.moving_reta)
        ema_restort=ema.variables_to_restore()
        saver=tf.train.Saver(ema_restort)
        #转换布尔值为小数
        cp=tf.equal(tf.argmax(pd,1),tf.argmax(y,1))
        acc=tf.reduce_mean(tf.cast(cp,tf.float32))
        while True:
            with tf.Session()as sess:
                ckpt=tf.train.get_checkpoint_state(mnist_backward_00.model_save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accs=sess.run(acc,feed_dict={x:mnist.test.images,y:mnist.test.labels})
                    print(global_step,accs)
                else:
                        print('No,model')
            time.sleep(5)
        
def main():
    mnist=read_data_sets('mnist_data',one_hot=True)
    test(mnist)

if __name__=='__main__':
    main()                
                
        

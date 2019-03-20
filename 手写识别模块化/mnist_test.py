# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:40:45 2019

@author: Administrator
"""

import tensorflow as tf
import mnist_backward
import mnist_farward
import time
#from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import mnist_tfrecord
batch_size=10000
def test():
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[None,mnist_farward.input_mode])
        y=tf.placeholder(tf.float32,[None,mnist_farward.output_mode])
        pd=mnist_farward.farward(x,None)
        #pd=tf.argmax(y,1)
        
        #滑动搞平均
        ema=tf.train.ExponentialMovingAverage(mnist_backward.moving_rate)
        ema_restord=ema.variables_to_restore()
        saver=tf.train.Saver(ema_restord)
        cp=tf.equal(tf.argmax(pd,1),tf.argmax(y,1))
        acc=tf.reduce_mean(tf.cast(cp,tf.float32))
        img_batch,labels_batch=mnist_tfrecord.Tfrecorde_load(batch_size,isTrain=False)
        while True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state(mnist_backward.model_save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    
                    coord=tf.train.Coordinator()
                    theads=tf.train.start_queue_runners(sess=sess,coord=coord)
                    
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    #accs=sess.run(acc,feed_dict={x:mnist.test.images,y:mnist.test.labels})
                    xs,ys=sess.run([img_batch,labels_batch])
                    accs=sess.run(acc,feed_dict={x:xs,y:ys})
                    print(global_step,accs)
                    coord.request_stop()
                    coord.join(theads)
                                 
                else:
                    print('没有训练包')
                    return
            time.sleep(5)

def main():
    #mnist=read_data_sets('mnist_data',one_hot=True)
    #test(mnist)
    test()
if __name__=='__main__':
    main()

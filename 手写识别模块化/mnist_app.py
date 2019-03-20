# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:51:06 2019

@author: Administrator
"""

import tensorflow as tf
import mnist_backward
import mnist_farward
import numpy as np
from PIL import Image

def image_search(image_data):
    with tf.Graph().as_default()as tg:
        x=tf.placeholder(tf.float32,[None,mnist_farward.input_mode])
        pd=mnist_farward.farward(x,None)
        loss=tf.argmax(pd,1)
        #滑动平均
        vt=tf.train.ExponentialMovingAverage(mnist_backward.moving_rate)
        vtr=vt.variables_to_restore()
        saver=tf.train.Saver(vtr)
        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state(mnist_backward.model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                loss_data=sess.run(loss,feed_dict={x:image_data})
                return loss_data

def image_load(image_op):
    image_open=Image.open(image_op)
    img_data=image_open.resize((28,28),Image.ANTIALIAS)
    img=np.array(img_data.convert('L'))
    thod=50
    for i in range(28):
        for j in range(28):
            img[i][j]=255-img[i][j]
            if (img[i][j]<thod):
                img[i][j]=0
            else:
                img[i][j]=255
    nm_arr=img.reshape([1,784])
    nm_arr.astype(np.float32)
    image_readly=np.multiply(nm_arr,1./255.)
    return image_readly
    
def apploction():
    image_index=int(input('输入测试图片数量：'))
    for i in range(image_index):
        image_op=input('图片地址')
        image_data=image_load(image_op)
        image_pp=image_search(image_data)
        print(image_pp)
    
def main():
    apploction()
if __name__=='__main__':
    main()

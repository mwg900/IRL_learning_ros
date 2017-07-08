#!/usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import rospy
from std_msgs.msg import Float32
from tensorflow.python.ops.variables import trainable_variables

def talker():
    pub = rospy.Publisher('loss', Float32, queue_size=10)
    rospy.init_node('linear_regression', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    
    #NN parameters
    LEARNING_RATE = 0.001
    NUM_POINTS = 1000
    #model_path = '../model/linear.ckpt'
    model_path = "/tmp/linear.ckpt"

        
    # 학습에 직접적으로 사용하지 않고 학습 횟수에 따라 단순히 증가시킬 변수 생성
    global_step = tf.Variable(0, trainable=False, name ='global_step')

    vectors_set = []
    for i in xrange(NUM_POINTS):
         x1= np.random.normal(0.0, 0.55)
         y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
         vectors_set.append([x1, y1])
    
    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]
    input_size = len(x_data)
    output_size = len(y_data)
    
    x_data = np.reshape(x_data, [1,input_size])
    y_data = np.reshape(y_data, [1,output_size])

    X = tf.placeholder(tf.float32, [1, input_size], name="input_x")
    l1 = tf.layers.dense(X, output_size, use_bias = True, bias_initializer = tf.zeros_initializer(), trainable = True)
    Y = l1
    
    loss = tf.reduce_mean(tf.square(Y - y_data))
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    #학습을 진행할 때마다 학습횟수 1씩 증가 
    train = optimizer.minimize(loss, global_step=global_step)
    
    
    
    #Traning
    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    #global_variables 함수를 통하여 앞서 정의하였던 변수들을 저장하거나 불러올 변수들로 설정
    #All variables 저장 
    saver = tf.train.Saver()
    #model 폴더로부터 불러올 Check point 정의
    #ckpt = tf.train.get_checkpoint_state('../model')
    
    #if tf.train.checkpoint_exists(model_path):
    #    saver.restore(sess, model_path)
        #print("restore comlete!")
    #else:
    sess.run(init)
    saver.restore(sess, model_path)
    print("restore comlete!")
    #sess.run(init)
    
    step=0
    while not rospy.is_shutdown():
        step += 1
        if step < 30:
            sess.run(train, feed_dict ={X : x_data})
            loss_value = sess.run(loss, feed_dict={X : x_data})
            print(step, loss_value)
        elif step == 30:
            #saver_path = saver.save(sess, model_path, global_step=global_step)
            saver.save(sess, model_path)
    
            print('variables saved!')
        pub.publish(loss_value)
    
        rate.sleep()
        rospy.is_shutdown()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

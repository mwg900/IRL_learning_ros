#!/usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import rospy
from std_msgs.msg import Float32

def talker():
    pub = rospy.Publisher('loss', Float32, queue_size=10)
    rospy.init_node('linear_regression', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    
    #NN parameters
    learning_rate = 0.001
    num_points = 1000
    vectors_set = []
    for i in xrange(num_points):
         x1= np.random.normal(0.0, 0.55)
         y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
         vectors_set.append([x1, y1])
    
    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]
    input_size = len(x_data)
    output_size = len(y_data)
    
    x_data = np.reshape(x_data, [1,input_size])
    y_data = np.reshape(y_data, [1,output_size])
    print(x_data)
    
    '''
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b
    '''
    
    X = tf.placeholder(tf.float32, [1, input_size], name="input_x")
    l1 = tf.layers.dense(X, output_size, use_bias = True, bias_initializer = tf.zeros_initializer())
    Y = l1
    
    loss = tf.reduce_mean(tf.square(Y - y_data))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    sess.run(init)
    
    step=0

    while not rospy.is_shutdown():
        step += 1
        sess.run(train, feed_dict ={X : x_data})
        loss_value = sess.run(loss, feed_dict={X : x_data})
        print(step, loss_value)
        pub.publish(loss_value)
        
        if step == 20:
            plt.plot(x_data, y_data, 'ro')
            plt.plot(x_data, sess.run(Y, feed_dict={X : x_data}))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(-2,2)
            plt.ylim(0.1,0.6)
            plt.show()
            
        rate.sleep()
        

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

'''
 #Graphic display
plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.xlabel('x')
plt.xlim(-2,2)
plt.ylim(0.1,0.6)
plt.ylabel('y')
plt.legend()
plt.show()
'''
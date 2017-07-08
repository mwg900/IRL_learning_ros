#!/usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

import rospy
from std_msgs.msg import Float32
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# Create model
def multilayer_perceptron(x):
    # Hidden layer with RELU activation
    layer_1 = tf.layers.dense(x,        n_hidden_1, activation=tf.nn.relu)
    
    # Hidden layer with RELU activation
    layer_2 = tf.layers.dense(layer_1,  n_hidden_2, activation=tf.nn.relu)  
    
    # Output layer with linear activation
    out_layer = tf.layers.dense(layer_2, n_classes)
    return out_layer
  
    
def talker():
    pub = rospy.Publisher('loss', Float32, queue_size=10)
    rospy.init_node('linear_regression', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    
    # Parameters
    learning_rate = 0.001
    batch_size = 100
    display_step = 1
    model_path = "model/model.ckpt"
    
    
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Construct model
    pred = multilayer_perceptron(x)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()

    while not rospy.is_shutdown():
        
        # Running first session
        '''
        
        print("Starting 1st session...")
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
        
            # Training cycle
            for epoch in range(3):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                                  y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost))
            print("First Optimization Finished!")
        
            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
            # Save model weights to disk
            save_path = saver.save(sess, model_path)
            print("Model saved in file: %s" % save_path)
        '''
        
        
        # Running a new session
        print("Starting 2nd session...")
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
            # Restore model weights from previously saved model
            saver.restore(sess, model_path)
            print("Model restored from file")
            
            '''
            
            # Resume training
            for epoch in range(7):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples / batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                                  y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                        "{:.9f}".format(avg_cost))
            saver.save(sess, model_path)
            print("Second Optimization Finished!")
            '''
            
            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Test from saved model")
            print("Accuracy:", accuracy.eval(
                {x: mnist.test.images, y: mnist.test.labels}))

        #pub.publish(avg_cost)
    
        #rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
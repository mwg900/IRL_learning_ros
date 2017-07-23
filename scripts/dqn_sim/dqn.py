#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""DQN Class
DQN(NIPS-2013)
"Playing Atari with Deep Reinforcement Learning"
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
DQN(Nature-2015)
"Human-level control through deep reinforcement learning"
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.python.slim import learning


class DQN:

    def __init__(self, session, input_size, output_size, name="main"):
        """DQN Agent can
        1) Build network
        2) Predict Q_value given state
        3) Train parameters
        Args:
            session (tf.Session): Tensorflow session
            input_size (int): Input dimension
            output_size (int): Number of discrete actions
            name (str, optional): TF Graph will be built under this name scope
        """
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        
        self._build_network()
        
        self.saver = tf.train.Saver()
        
    def _build_network(self, h_size=16, l_rate=0.001):
        """DQN Network architecture (simple MLP)
        Args:
            h_size (int, optional): Hidden layer dimension
            l_rate (float, optional): Learning rate
        """
        with tf.variable_scope(self.net_name):
            #with tnesorflow 1.2
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            l1 = tf.layers.dense(self._X, h_size, activation=tf.nn.relu)
            l2 = tf.layers.dense(l1, self.output_size)
            self._Qpred = l2

            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)

            optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self._train = optimizer.minimize(self._loss)
          

            
    # state에 따라 Q 함수의 값을 돌려주는 함수
    def predict(self, state):
        """Returns Q(s, a)
        Args:
            state (np.ndarray): State array, shape (n, input_dim)
        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """
        x = np.reshape(state, [-1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    # X, Y 입력만으로 트레이닝 업데이트를 해주는 함수
    def update(self, x_stack, y_stack): 
        """Performs updates on given X and y and returns a result
        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)
        Returns:
            list: First element is loss, second element is a result from train step
        """
        return self.session.run([self._loss, self._train], feed_dict = {self._X: x_stack, self._Y: y_stack})
    
    
    
    def save_data(self, sess, model_path):
        # Save model weights to disk
        self.saver.save(sess, model_path)
        
    
    def restore_data(self, sess, model_path):
        # restore model weights from disk
        self.saver.restore(sess, model_path)
        
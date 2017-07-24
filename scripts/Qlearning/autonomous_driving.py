#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Ros node for Autonomous driving using Q-learning
Edited by Jeong-Hwan Moon, IRL, Pusan National UNIV. mwg900@naver.com
""" 

import numpy as np
import tensorflow as tf
import rospy
import dqn


from IRL_learning_ros.msg import State
from std_msgs.msg import Int8
from argparse import Action



INPUT_SIZE = 9                       # [-60, -45, -30, -15, 0, 15, 30, 45, 60] 
OUTPUT_SIZE = 5                      # [전진, 좌회전, 우회전]


class state_pub:
    def __init__(self): 
        node_name = "state_pub" 
        rospy.init_node(node_name)
        #ros topic 구독자 설정 및 콜백함수 정의
        state_sub = rospy.Subscriber("/state", State, self.state_callback, queue_size=100)
        self.pub = rospy.Publisher('/action', Int8, queue_size=10)
        self.rate = rospy.Rate(5) # 10hz
        self.F = False 
        
        
    #Laser 토픽 콜백
    def state_callback(self, msg):
        self.state = msg.ranges
        self.done = msg.done
        self.F = True
    

    def talker(self):
        with tf.Session() as sess:
            mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE)    #DQN class 선언
            init = tf.global_variables_initializer()
            sess.run(init)
            
            while not rospy.is_shutdown():                      #루프 실행
                if self.F is True:
                    state = self.state          # 거리정보 복사
                    action = np.argmax(mainDQN.predict(state))  #거리에 따른 액션 값 획득(로봇 행동 지령)
                    self.pub.publish(action)            #액션 값 퍼블리시
                    self.rate.sleep()                   #ROS sleep
            
if __name__ == '__main__':
    try:
        main = state_pub()
        main.talker()
    except rospy.ROSInterruptException:
        pass
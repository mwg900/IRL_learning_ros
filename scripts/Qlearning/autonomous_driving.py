#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Ros node for Autonomous driving using Q-learning
Edited by Jeong-Hwan Moon, IRL, Pusan National UNIV. mwg900@naver.com
""" 

import numpy as np
import tensorflow as tf
import random
import rospy
import dqn
import policy
import register
import sys

from collections import deque

from IRL_learning_ros.msg import State
from std_msgs.msg import Int8
from argparse import Action


#Hyper parameter
ENVIRONMENT = rospy.get_param('/autonomous_driving/environment', 'v1')
MODEL_PATH = rospy.get_param('/autonomous_driving/model_path', default = 'model')
MODEL_DATA = rospy.get_param('/autonomous_driving/model_data', default = '1620')

if ENVIRONMENT == 'v0':
    INPUT_SIZE =  register.environment.v0['input_size']
    OUTPUT_SIZE = register.environment.v0['output_size']
    POLICY =      register.environment.v0['policy']
    print('Autonomous_driving training v0 is ready')
    print(register.environment.v0)

elif ENVIRONMENT == 'v1':
    INPUT_SIZE =  register.environment.v1['input_size']
    OUTPUT_SIZE = register.environment.v1['output_size']
    POLICY =      register.environment.v1['policy']
    print('Autonomous_driving training v1 is ready')
    print(register.environment.v1)

else:
    print("E: you select wrong environment. you must select ex) env:=v1 or env:=v0")
    sys.exit()



class state_pub:
    def __init__(self): 
        node_name = "state_pub" 
        rospy.init_node(node_name)
        #ros topic 구독자 설정 및 콜백함수 정의
        state_sub = rospy.Subscriber("/state", State, self.state_callback, queue_size=100)
        self.pub = rospy.Publisher('/action', Int8, queue_size=10)
        self.rate = rospy.Rate(5) # 10hz
        self.F = False 
        self.load_epi = str(MODEL_DATA)
        self.model_path = MODEL_PATH +"/confirmed_model/"+ENVIRONMENT+"/"+self.load_epi+"/driving15"+ENVIRONMENT+".ckpt"
        
    #Laser 토픽 콜백
    def state_callback(self, msg):
        self.state = msg.ranges
        self.done = msg.done
        self.F = True
    

    def talker(self):
        with tf.Session() as sess:
            mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name = "main")    #DQN class 선언
            #targetDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name ="target")
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(max_to_keep= 5)
            sess.run(init)
            #------------------------------------------------------------------------------ 
            # Model load
            #------------------------------------------------------------------------------ 
            model_name = self.model_path+"-"+self.load_epi
            saver.restore(sess, model_name)            #저장된 데이터 불러오기
            print("Model restored from {}".format(model_name))
            #------------------------------------------------------------------------------ 
            reward_sum = 0
            reward = 0
            while not rospy.is_shutdown():                      #루프 실행
                if self.F is True:
                    state = self.state          # 거리정보 복사
                    done = self.done
                        # Reward Policy
                    try:
                        if POLICY == 'autonomous_driving':
                            reward = policy.autonomous_driving(action, done)     #reward 리턴
                        elif POLICY == 'autonomous_driving1':
                            reward = policy.autonomous_driving1(action, done)     #reward 리턴
                    except:
                        print('there is no policy') 
                    action = np.argmax(mainDQN.predict(state, 1.0))  #거리에 따른 액션 값 획득(로봇 행동 지령)
                    self.pub.publish(action)            #액션 값 퍼블리시
                    
                    reward_sum += reward
                    print("action : {:>5}, current score : {:>5}".format(action, reward_sum))   
                    if done == True:
                        reward_sum = 0                
                    self.rate.sleep()                   #ROS sleep
            
if __name__ == '__main__':
    try:
        main = state_pub()
        main.talker()
    except rospy.ROSInterruptException:
        pass
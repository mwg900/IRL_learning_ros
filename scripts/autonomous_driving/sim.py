#!/usr/bin/env python 
#-*- coding: utf-8 -*- 
"""
Ros node for Gazebo simulation model spawn
Edited by Jeong-Hwan Moon, IRL, Pusan National UNIV. mwg900@naver.com
""" 

import rospy
import random
import tf
#import service messages
from IRL_learning_ros.srv import SpawnPos
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty



class simulation_server:
    def __init__(self): 
        node_name = "sim_server" 
        rospy.init_node(node_name)
        rospy.Service('model_respawn', SpawnPos, self.model_respawn)
        self.reset   = rospy.ServiceProxy('gazebo/reset_world', Empty)            # World reset 서비스 요청 함수 선언
        self.set_pos = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)# Model state set 서비스 요청 함수 선언
        print ("Ready to spwan the model.")
        
    
    def rand_pos(self):    
        #x = random.uniform(-10.0, 10.0)
        #y = random.uniform(-10.0, 10.0)
        x = 0
        y = 0
        theta = random.uniform(-3.1415, 3.1415)
        return x, y, theta
        
                
    # spawn service 콜백 함수 (from spawn req)
    def model_respawn(self, srv):
        #res 요청이 들어왔을 시 실행
        x, y, theta = self.rand_pos()                  # randomly select spawn positon
        
        state = ModelState()
        state.model_name = 'agent'
        state.pose.position.x = x
        state.pose.position.y = y
        
        quaternion = tf.transformations.quaternion_from_euler(0.0,0.0,theta)            #롤, 피치, 요우로부터 쿼터니언 값을 얻음 
        
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        
        self.set_pos(state)
        self.reset()
        self.set_pos(state)     #버그 방지를 위해 모델 리스폰 -> 월드 리셋 -> 모델 리스폰 순으로 정리 
        return x, y, theta

if __name__ == "__main__":
    try:
        main = simulation_server()
        rospy.spin()
    except rospy.ROSInterruptException: pass
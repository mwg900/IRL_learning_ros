#!/usr/bin/env python 
#-*- coding: utf-8 -*- 

import os
import rospy
import random
#import service messages
from IRL_learning_ros.srv import SpawnPos
from gazebo_msgs.srv import DeleteModel

class simulation_server:
    def __init__(self): 
        node_name = "sim_server" 
        rospy.init_node(node_name)
        s = rospy.Service('model_respawn', SpawnPos, self.model_respawn)
        print ("Ready to spwan the model.")
        


        #Kobuki 생성
        #rosrun gazebo_ros spawn_model -file `rospack find IRL_learning_ros`/urdf/kobuki_hexagons_asus_rplidar.urdf.xacro -urdf -y 1 -model rrbot1 -robot_namespace kobuki
        kobuki_description = "`rospack find IRL_learning_ros`/urdf/kobuki_hexagons_asus_rplidar.urdf.xacro"
        arm_description = "`rospack find irl_dual_arm`/urdf/irl_dual_arm.urdf"
        agent_name = 'IRL_kobuki'
        x = 0
        y = 0
        commend = "rosrun gazebo_ros spawn_model -unpause -urdf -file {} -model {} -x {} -y {} -Y 0 robot_namespace {}".format(kobuki_description, agent_name, x, y, agent_name)
        #commend = "rosrun gazebo_ros spawn_model -urdf -file {} -model {} -x {} -y {} ".format(arm_description, 'arm', x, y)
        print(commend)
        os.system(commend)
    #randomly selected spawn positon
    def rand_pos(self):    
        x = random.uniform(-10.0, 10.0)
        y = random.uniform(-10.0, 10.0)
        return x, y
        
    #delete service 요청 함수 (to gazebo)
    def delete_model(self, model_name):
        rospy.wait_for_service('gazebo/delete_model')
        d = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        d(model_name)
        
    #spawn service 콜백 함수 (from spawn req)
    def model_respawn(self, srv):
        model_name = 'coke_can'
        
        #res 요청이 들어왔을 시 실행
        self.delete_model(model_name)           # delete existing model
        x, y = self.rand_pos()                  # randomly selected spawn positon
        os.system("rosrun gazebo_ros spawn_model -database coke_can -gazebo -model {} -z 5 -x {} -y {}".format(model_name, x, y))     #System 명령어 호출 = "rosrun gazebo spawn_model -file <path to xml/urdf file> -urdf -model <model_name>"
        return model_name, x, y 

if __name__ == "__main__":
    try:
        main = simulation_server()
        rospy.spin()
    except rospy.ROSInterruptException: pass
#!/usr/bin/env python
#-*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
from IRL_learning_ros.msg import State 
from IRL_learning_ros.srv import SpawnPos

class state_pub:
    def __init__(self): 
        node_name = "state_pub" 
        rospy.init_node(node_name)
        #ros topic 구독자 설정 및 콜백함수 정의
        laser_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size=100)
        self.respawn = rospy.ServiceProxy('/model_respawn', SpawnPos)   # Model state set 서비스 요청 함수 선언
        self.pub = rospy.Publisher('/state', State, queue_size=10)
        self.rate = rospy.Rate(10) # 10hz
        self.F = False
    #Laser 토픽 콜백
    def scan_callback(self, LaserScan):
        self.F = True
        #topic 복사
        self.done = False
        self.range_state = []
        for ang in range(115,256,15):    #-75도 ~ 75도를 15도 간격마다 확인
            self.range_state.append(LaserScan.ranges[ang])
            if LaserScan.ranges[ang] < 0.21:
                self.respawn()
                self.done = True

    def talker(self):
        while not rospy.is_shutdown():
            if self.F is True:
                msg = State()
                msg.ranges = self.range_state
                msg.done = self.done
                self.pub.publish(msg)
            self.rate.sleep()
     
if __name__ == '__main__':
    try:
        main = state_pub()
        main.talker()
    except rospy.ROSInterruptException:
        pass
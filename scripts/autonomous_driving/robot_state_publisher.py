#!/usr/bin/env python
#-*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

class state_pub:
    def __init__(self): 
        node_name = "state_pub" 
        rospy.init_node(node_name)
        #ros topic 구독자 설정 및 콜백함수 정의
        laser_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size= 100)
        self.pub = rospy.Publisher('/state', Float32, queue_size=10)
        self.rate = rospy.Rate(10) # 10hz
        
        
    #Laser 토픽 콜백
    def scan_callback(self, LaserScan):
        #topic 복사
        self.scan_msg = LaserScan
        self.F = True
        
    
    def talker():
        while not rospy.is_shutdown():
            #pub.publish(avg_cost)
            self.rate.sleep()
    
if __name__ == '__main__':
    try:
        main = state_pub()
        main.talker()
    except rospy.ROSInterruptException:
        pass
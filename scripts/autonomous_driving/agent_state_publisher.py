#!/usr/bin/env python
#-*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

class state_pub:
    def __init__(self): 
        node_name = "state_pub" 
        rospy.init_node(node_name)
        #ros topic 구독자 설정 및 콜백함수 정의
        laser_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size= 100)
        self.pub = rospy.Publisher('/state', Float32MultiArray, queue_size=10)
        self.rate = rospy.Rate(10) # 10hz
        self.F = False
        
    #Laser 토픽 콜백
    def scan_callback(self, LaserScan):
        #topic 복사
        self.scan_msg = LaserScan
        print(self.scan.ranges[100:105])
        self.F = True
        
    
    def talker(self):
        while not rospy.is_shutdown():
            if self.F is True:
                msg = Float32MultiArray()
                msg.data = self.scan.ranges
                pub.publish(msg)
            #self.rate.sleep()
     
if __name__ == '__main__':
    try:
        main = state_pub()
        main.talker()
    except rospy.ROSInterruptException:
        pass
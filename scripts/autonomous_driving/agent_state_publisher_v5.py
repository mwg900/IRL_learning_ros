#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Ros node for RPlidar sensor msg pub
Edited by Jeong-Hwan Moon, IRL, Pusan National UNIV. mwg900@naver.com
""" 
import rospy
from sensor_msgs.msg import LaserScan
from IRL_learning_ros.msg import State 
#from IRL_learning_ros.srv import SpawnPos

INVERSED = rospy.get_param('/agent_state_publisher/inversed', True)
#INVERSED = False
class state_pub:
    def __init__(self): 
        node_name = "state_pub" 
        rospy.init_node(node_name)
        #ros topic 구독자 설정 및 콜백함수 정의
        laser_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size=100)
        self.pub = rospy.Publisher('/state', State, queue_size=10)
        self.rate = rospy.Rate(20) # 5hz
        self.F = False
        self.noise_count = 0
        self.angle =  [95, 120, 135, 150, 165, 180, 195, 210, 225, 240, 265] 
        self.notinf = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        if INVERSED == True:
            self.angle = [88, 60, 45, 30, 15, 0, 345, 330, 315, 300, 272]
            
            
    #Laser 토픽 콜백
    def scan_callback(self, LaserScan):
        #topic 복사
        self.done = False
        tmp_state = []
        ang_cnt = 0
        for ang in self.angle:    #-75도 ~ 75도를 15도 간격마다 확인        [95, 120, 135, 150, 165, 180, 195, 210, 225, 240, 265] 
            if (LaserScan.ranges[ang] != 0):
                tmp_state.append(LaserScan.ranges[ang])
            else:
                self.noise_count+=1
                if self.noise_count > 10:
                    tmp_state.append(6.0)           #0이 아닌 6.0미터가 입력되도록 함
                    self.noise_count = 0
                else:
                    tmp_state.append(self.notinf[ang_cnt]) 
            if (ang == 265) or (ang == 272):
                self.range_state = tmp_state
                self.notinf = tmp_state
            if (LaserScan.ranges[ang] != 0) and (LaserScan.ranges[ang] < 0.21):        # 인식 거리가 21cm 미만일 시 (충돌) 완료플래그 셋
                self.done = True
            ang_cnt +=1
        self.F = True
        
    def talker(self):
        while not rospy.is_shutdown():
            if self.F is True:
                msg = State()
                msg.ranges = self.range_state
                msg.done = self.done  
                #print(msg.ranges[7])            # 7 = 가제보 : 시계방향 30도 rplidar : 반시계방향 30도 
                self.pub.publish(msg)
                self.rate.sleep()
     
if __name__ == '__main__':
    try:
        main = state_pub()
        main.talker()
    except rospy.ROSInterruptException:
        pass
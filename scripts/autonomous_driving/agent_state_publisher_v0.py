#!/usr/bin/env python
#-*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
from IRL_learning_ros.msg import State 
#from IRL_learning_ros.srv import SpawnPos

INVERSED = rospy.get_param('/agent_state_publisher/inversed', False)
class state_pub:
    def __init__(self): 
        node_name = "state_pub" 
        rospy.init_node(node_name)
        #ros topic 구독자 설정 및 콜백함수 정의
        laser_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size=100)
        self.pub = rospy.Publisher('/state', State, queue_size=10)
        #self.rate = rospy.Rate(10) # 10hz
        self.F = False
        self.angle = [95, 120, 135, 150, 165, 180, 195, 210, 225, 240, 265] 
        if INVERSED == True:
            self.angle = [275, ]
    #Laser 토픽 콜백
    def scan_callback(self, LaserScan):
        
        #topic 복사
        self.done = False
         
        tmp_state = []
        
        if INVERSED == False:
            for ang in range(120,241,15):    #-75도 ~ 75도를 15도 간격마다 확인        [95, 120, 135, 150, 165, 180, 195, 210, 225, 240, 265] 
                if ang == 120:
                    tmp_state.append(LaserScan.ranges[95])
                tmp_state.append(LaserScan.ranges[ang])
                if ang == 240:               # 루프문이 끝날 시에만 값 할당
                    tmp_state.append(LaserScan.ranges[265])
                    self.range_state = tmp_state
                    
                if LaserScan.ranges[ang] < 0.21:        # 인식 거리가 21cm 미만일 시 (충돌) 완료플래그 셋
                    self.done = True
                
        elif INVERSED == True:
            for ang in range(INV_ANGLE-240, INV_ANGLE-119, 15):    #-75도 ~ 75도를 15도 간격마다 확인        [95, 120, 135, 150, 165, 180, 195, 210, 225, 240, 265] 
                if ang == (INV_ANGLE-240):
                    tmp_state.append(LaserScan.ranges[INV_ANGLE-265])
                tmp_state.append(LaserScan.ranges[ang])
                if ang == (INV_ANGLE-119):               # 루프문이 끝날 시에만 값 할당
                    tmp_state.append(LaserScan.ranges[INV_ANGLE-95])
                    self.range_state = tmp_state
                    
                if LaserScan.ranges[ang] < 0.21:        # 인식 거리가 21cm 미만일 시 (충돌) 완료플래그 셋
                    self.done = True
        self.F = True
        
    def talker(self):
        while not rospy.is_shutdown():
            if self.F is True:
                msg = State()
                msg.ranges = self.range_state
                msg.done = self.done
                if self.done == True:
                    rospy.wait_for_service('gazebo/reset_world')    #reset 될 때까지 대기
                    rospy.sleep(0.1)    #0.1초동안 딜레이
                self.pub.publish(msg)
            #self.rate.sleep()
     
if __name__ == '__main__':
    try:
        main = state_pub()
        main.talker()
    except rospy.ROSInterruptException:
        pass
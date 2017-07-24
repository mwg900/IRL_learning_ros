#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Ros node for Kobuki motor effort pub
Edited by Jeong-Hwan Moon, IRL, Pusan National UNIV. mwg900@naver.com
""" 
import rospy
from std_msgs.msg import Int8
from geometry_msgs.msg import Twist     #속도제어용 메세지 
from kobuki_msgs.msg import MotorPower  #모터 ON/OFF 메세지
from sensor_msgs.msg import LaserScan   #라이다 데이터 메세지


class state_pub:
    def __init__(self): 
        node_name = "effort_pub" 
        rospy.init_node(node_name)
        #ros topic 구독자 설정 및 콜백함수 정의
        act_sub = rospy.Subscriber("/action", Int8, self.action_callback, queue_size=100)
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
        self.motor_pub = rospy.Publisher('/mobile_base/commands/motor_power', MotorPower, queue_size=10)
        
        self.rate = rospy.Rate(10) # 10hz
        self.vel = Twist()          #Twist() 메세지 형태로 vel 변수 선언

    #Laser 토픽 콜백
    def action_callback(self, msg):
        vel = Twist()
        if msg.data == 0: #직진
            vel.linear.x = 0.9
        elif msg.data == 1: #우회전
            vel.linear.x = 0.7
            vel.angular.z = -1.2
        elif msg.data == 2: #좌회전
            vel.linear.x = 0.7
            vel.angular.z = 1.2
        elif msg.data == 3: #제자리우회전
            vel.linear.x = 0.0
            vel.angular.z = -2.0
        elif msg.data == 4: #제자리좌회전
            vel.linear.x = 0.0
            vel.angular.z = 2.0
            
        elif msg.data == 99: # 정지(리셋)
            self.motor_pub.publish(MotorPower.OFF)
            self.motor_pub.publish(MotorPower.ON)
        self.vel_pub.publish(vel)   #속도값 퍼블리시
        
if __name__ == '__main__':
    try:
        main = state_pub()
        rospy.spin()        #루프 회전
    except rospy.ROSInterruptException:
        pass
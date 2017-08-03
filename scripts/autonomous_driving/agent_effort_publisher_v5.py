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

STRAIGHT_VEL=0.5
TURNING_VEL =0.3

TURNING_ANG= 0.6
ROTATE_ANG = 1.0

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
        self.count = 6
        self.pre_act = 3
        self.current_vel = 0.0
        self.current_ang = 0.0

        
    def vel_control(self, action):
        x = 0.0
        z = 0.0
        if action == 0:             #전진
            if self.pre_act != 0:
                self.count = 6
            self.count -= 1
            if self.count == 0: 
                self.count = 1                
            self.pre_act = 0
            # Curruent_velocity + (Goal_velocity Current_velocity)/count
            x = self.current_vel + (STRAIGHT_VEL - self.current_vel)/self.count
            z = self.current_ang + (0.0 - self.current_ang)/self.count
            self.current_vel = x
            self.current_ang = z
            
        elif action == 1:        #좌회전
            if self.pre_act != 1:
                self.count = 5
                self.current_ang = 0.0
            self.count -= 1
            if self.count == 0: 
                self.count = 1                
            self.pre_act = 1
            # Curruent_velocity + (Goal_velocity Current_velocity)/count
            x = self.current_vel + (TURNING_VEL - self.current_vel)/self.count
            z = self.current_ang + (-TURNING_ANG - self.current_ang)/self.count
            self.curent_vel = x
            self.current_ang = z
            
        elif action == 2:       #우회전
            if self.pre_act != 2:
                self.count = 5
                self.current_ang = 0.0
            self.count -= 1
            if self.count == 0: 
                self.count = 1                
            self.pre_act = 2
            # Curruent_velocity + (Goal_velocity Current_velocity)/count
            x = self.current_vel + (TURNING_VEL - self.current_vel)/self.count
            z = self.current_ang + (TURNING_ANG - self.current_ang)/self.count
            self.curent_vel = x
            self.current_ang = z

        elif action == 3:     #제자리 좌회전
            if self.pre_act != 3:
                self.count = 4
                self.current_ang = 0.0
            self.count -= 1
            if self.count == 0: 
                self.count = 1                
            self.pre_act = 3
            x = 0.0
            z = self.current_ang + (-ROTATE_ANG - self.current_ang)/self.count
            self.curent_vel = x
            self.current_ang = z

        elif action == 4:      #제자리 우회전
            if self.pre_act != 4:
                self.count = 4
                self.current_ang = 0.0
            self.count -= 1
            if self.count == 0: 
                self.count = 1                
            self.pre_act = 4
            x = 0.0
            z = self.current_ang + (ROTATE_ANG - self.current_ang)/self.count
            self.curent_vel = x
            self.current_ang = z
        return x, z
    
    
    #Laser 토픽 콜백
    def action_callback(self, msg):
        vel = Twist()

        if msg.data == 99: # 정지(리셋)
            vel.linear.x = 0.0
            vel.angular.z = 0.0
            self.motor_pub.publish(MotorPower.OFF)
            self.motor_pub.publish(MotorPower.ON)
        else:
            vel.linear.x, vel.angular.z = self.vel_control(msg.data)
        #print(msg.data, vel)
        self.vel_pub.publish(vel)   #속도값 퍼블리시
        
if __name__ == '__main__':
    try:
        main = state_pub()
        rospy.spin()        #루프 회전
    except rospy.ROSInterruptException:
        pass
#!/usr/bin/env python
#-*- coding: utf-8 -*-

import rospy
from IRL_learning_ros.msg import Action
from geometry_msgs.msg import Twist     #속도제어용 메세지 
from kobuki_msgs.msg import MotorPower  #모터 ON/OFF 메세지
from sensor_msgs.msg import LaserScan   #라이다 데이터 메세지


class state_pub:
    def __init__(self): 
        node_name = "effort_pub" 
        rospy.init_node(node_name)
        #ros topic 구독자 설정 및 콜백함수 정의
        act_sub = rospy.Subscriber("/action", Action, self.action_callback, queue_size=100)
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
        #self.motor_pub = rospy.Publisher('/mobile_base/commands/motor_power', MotorPower, queue_size=10)
        
        self.rate = rospy.Rate(10) # 10hz
        self.vel = Twist()          #Twist() 메세지 형태로 vel 변수 선언
        
        
    
    #Laser 토픽 콜백
    def action_callback(self, msg):
        #self.motor_pub(MotorPower.ON)
        vel = Twist()
        if msg.action == 1: #직진
            vel.linear.x = 0.3
            vel.angular.z = 0
            print('go straight')
        elif msg.action == 2: #우회전
            vel.linear.x = 0.2
            vel.angular.z = -0.8
            print('turn right')
        elif msg.action == 3: #좌회전
            vel.linear.x = 0.2
            vel.angular.z = 0.8
            print('turn left')
        self.vel_pub.publish(vel)   #속도값 퍼블리시
        
if __name__ == '__main__':
    try:
        main = state_pub()
        rospy.spin()        #루프 회전
    except rospy.ROSInterruptException:
        pass
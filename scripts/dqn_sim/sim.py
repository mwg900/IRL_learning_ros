#!/usr/bin/env python 
#-*- coding: utf-8 -*- 

import sys
import rospy
from IRL_learning_ros.srv import SpawnPos

def add_two_ints_client(x, y):
    rospy.wait_for_service('add_two_ints')
    try:
        add_two_ints = rospy.ServiceProxy('add_two_ints', AddTwoInts)
        resp1 = add_two_ints(x, y)
        return resp1.sum
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def usage():
    return "%s [x y]"%sys.argv[0]

if __name__ == "__main__":
    if len(sys.argv) == 3:
        x = int(sys.argv[1])
        y = int(sys.argv[2])
    else:
        print usage()
        sys.exit(1)
    print "Requesting %s+%s"%(x, y)
    print "%s + %s = %s"%(x, y, add_two_ints_client(x, y))
    
    

'''
import rospy 
from visualization_msgs.msg import Marker
from robot_mapping.msg import Slavepos
#from robot_mapping.srv import Position, PositionResponse 
import math 
 
#클래스 시작 
class robot_marker: 
    def __init__(self): 
        node_name = "sim" 
        rospy.init_node(node_name, anonymous=True) #노드 초기화. 노드명은 1개만 가능
        position_sub = rospy.Subscriber("/position", Slavepos, self.callback, queue_size= 100)  #포지션 메세지 서브스크라이버 선언
        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)    #Marker 메세지 사용하는 viualization_marker 토픽 퍼플리셔 설정  

    def callback(self, msg):
        pass
        

 
if __name__ == '__main__': 
    try: 
        main = robot_marker()
        rospy.spin()                    #루프 회전 
    except rospy.ROSInterruptException: pass
    
    
    
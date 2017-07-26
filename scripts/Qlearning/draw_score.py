#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Ros node for Kobuki motor effort pub
Edited by Jeong-Hwan Moon, IRL, Pusan National UNIV. mwg900@naver.com
""" 
import rospy
import matplotlib.pyplot as plt
import numpy as np
import csv 


ENVIRONMENT = rospy.get_param('/draw_score/environment', 'v1')
MODEL_PATH = rospy.get_param('/draw_score/model_path',
                              default = '/home/moon/catkin_ws/src/IRL_learning_ros/IRL_learning_ros/model/result')

class draw_plot:
    def __init__(self): 
        node_name = "draw_plot" 
        rospy.init_node(node_name)
        #ros topic 구독자 설정 및 콜백함수 정의
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        self.rate = rospy.Rate(1) # 10hz
        x, y = self.load_file()
        self.li, = ax.plot(x, y)
        
        ax.relim() 
        ax.autoscale_view(True,True,True)
        self.fig.canvas.draw()
        plt.xlabel("episode")
        plt.ylabel("score")
        plt.show(block=False)
        plt.plot([1,5000],[1000,1000], 'k-')
    #Laser 토픽 콜백
    
    def load_file(self):
        with open(MODEL_PATH+'/'+ENVIRONMENT+'/'+'score.csv', 'rb') as csvfile: 
            x = []
            y = []
            
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                x.append(row[0])
                y.append(row[1])
    
            x = np.array(x)
            y = np.array(y)
            return x, y
        
    def drawer(self):
        while not rospy.is_shutdown():

            x, y = self.load_file()
            # set the new data
            self.li.set_xdata(x)
            self.li.set_ydata(y)

            self.fig.canvas.draw()
            self.rate.sleep()
        
if __name__ == '__main__':
    try:
        main = draw_plot()
        main.drawer()        #루프 회전
    except rospy.ROSInterruptException:
        pass
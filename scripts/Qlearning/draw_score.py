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
                              default = '/home/moon/catkin_ws/src/IRL_learning_ros/model/result')

class draw_plot:
    def __init__(self): 
        node_name = "draw_plot" 
        rospy.init_node(node_name)
        #ros topic 구독자 설정 및 콜백함수 정의
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.rate = rospy.Rate(1) # 10hz
        x, y, cnt = self.load_file()
        print(cnt)
        axes = plt.gca()
        axes.set_xlim([0,cnt+500])
        axes.set_ylim([-500,1200])
        self.li, = self.ax.plot(x, y)
        
        self.ax.relim() 
        self.ax.autoscale_view(True,True,True)
        plt.xlabel("episode")
        plt.ylabel("score")

        plt.show(block=False)
        
    #Laser 토픽 콜백
    
    def load_file(self):
        with open(MODEL_PATH+'/'+ENVIRONMENT+'/'+'score.csv', 'rb') as csvfile: 
            x = []
            y = []
            reader = csv.reader(csvfile, delimiter=',')
            
            for row in reader:
                x.append(int(row[0]))
                y.append(float(row[1]))
            row_count = max(x)  
            x = np.array(x)
            y = np.array(y)
            
        return x, y, row_count
        
    def drawer(self):
        while not rospy.is_shutdown():

            x, y, cnt = self.load_file()
            # set the new data
            self.li.set_xdata(x)
            self.li.set_ydata(y)
            plt.plot([1,cnt+500],[1000,1000], 'k-')
    
            self.fig.canvas.draw()
            self.rate.sleep()
        
if __name__ == '__main__':
    try:
        main = draw_plot()
        main.drawer()        #루프 회전
    except rospy.ROSInterruptException:
        pass

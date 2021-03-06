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

ENVIRONMENT = rospy.get_param('/draw_result/environment', 'v2')
MODEL_PATH = rospy.get_param('/draw_result/model_path', default = '/home/moon/catkin_ws/src/IRL_learning_ros/model/result')

class draw_plot:
    def __init__(self): 
        node_name = "draw_plot" 
        rospy.init_node(node_name)

        x, y1, y2, cnt, y3, y4 = self.load_file()
        #------------------------------------------------------------------------------ 
        # Figure 1
        #------------------------------------------------------------------------------ 
        self.cost = plt.figure(1,figsize=(7,10))
        self.cost_ax = self.cost.add_subplot(2,1,1)
        self.cost_ax.set_xlabel('episode')
        self.cost_ax.set_ylabel('cost')
        axes_1 = self.cost.gca()
        #cnt = 0
        #axes_1.set_xlim([0,cnt+500])    #컨퍼용 수정
        axes_1.set_xlim([0,660])
        axes_1.set_ylim([0,50])
        
        self.score_ax = self.cost.add_subplot(2,1,2)
        self.rate = rospy.Rate(1) # 10hz
        self.F = False
        
        
    
        #------------------------------------------------------------------------------ 
        # Figure 2
        #------------------------------------------------------------------------------ 
        axes = plt.gca()
        #axes.set_xlim([0,cnt+500])
        axes.set_xlim([0,1000])
        axes.set_ylim([-200,2000])
        self.li_1, = self.score_ax.plot(x, y1)
        self.li_2, = self.cost_ax.plot(x, y2)
        self.li_3, = self.cost_ax.plot(x, y3)
        self.li_4, = self.score_ax.plot(x, y4)
        #self.ax.relim() 
        #self.ax.autoscale_view(True,True,True)
        plt.xlabel("episode")
        plt.ylabel("score")
        plt.tight_layout()
        plt.show(block=False)
        
        print(cnt)
    #Laser 토픽 콜백
    
    def load_file(self):
        x = []
        y1 = []
        y2 = []
        y3 = []
        y4 = []
        episode_count = 0
        with open(MODEL_PATH+'/'+ENVIRONMENT+'/'+'score.csv', 'rb') as csvfile: 
            
            reader = csv.reader(csvfile, delimiter=',')
            row_count = sum(1 for row in reader)
        epi = 1
        if row_count > 1:
            with open(MODEL_PATH+'/'+ENVIRONMENT+'/'+'score.csv', 'rb') as csvfile: 
                reader = csv.reader(csvfile, delimiter=',')
                if row_count > 1:
                    for row in reader:
                        x.append(int(row[0]))
                        y1.append(float(row[1]))
                        y2.append(float(row[2]))
                        
                        min_cost = min(y2[-100:])
                        
                        avr = sum(y1[-20:])/20  #최근 20개의 score 평균치
                        #if avr > 1000 and self.F == False:
                        #    print(row[0])
                        #    self.F = True
                        
                        y3.append(min_cost)
                        y4.append(avr)
                        epi+=1
                    episode_count = max(x)
                    
                    
                    x = np.array(x)
                    y1 = np.array(y1)
                    y2= np.array(y2)
                    y3 = np.array(y3)            #평균
                    y4 = np.array(y4)
                    
        return x, y1, y2 ,episode_count, y3, y4
        
    def drawer(self):
        while not rospy.is_shutdown():

            x, y1, y2,cnt,y3, y4 = self.load_file()
            plt.plot([1,cnt+500],[1000,1000], 'k-', linestyle='--')
            # set the new data
            self.li_1.set_xdata(x)
            self.li_1.set_ydata(y1)
            self.li_2.set_xdata(x)
            self.li_2.set_ydata(y2)
            
            self.cost.canvas.draw()
            self.li_3.set_color('red')
            self.li_3.set_xdata(x)
            self.li_3.set_ydata(y3)
            
            self.li_4.set_color('red')
            self.li_4.set_xdata(x)
            self.li_4.set_ydata(y4)
            self.cost.canvas.draw()
            self.rate.sleep()
        
if __name__ == '__main__':
    try:
        main = draw_plot()
        main.drawer()        #루프 회전
    except rospy.ROSInterruptException:
        pass

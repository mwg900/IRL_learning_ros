#!/usr/bin/env python
#-*- coding: utf-8 -*-

class environment :
        
    def autonomous_driving(self, action, done):
        # Reward Policy
        if action == 0:
            reward = 5
        elif (action == 1) or (action == 2):
            reward = -0.5
        elif (action == 3) or (action == 4):
            reward = -20
        if done:            # 충돌  
            reward = -200
    
        return reward

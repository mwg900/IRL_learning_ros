#!/usr/bin/env python
#-*- coding: utf-8 -*-

class policy :
        
    def autonomous_driving(self, action, done):
        # Reward Policy
        if action == 0:
            reward = 5
        elif (action == 1) or (action == 2):
            reward = -0.5
        if done:            # 충돌  
            reward = -200
    
        return reward
    
    
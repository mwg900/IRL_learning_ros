#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Policy register for q-learning
Edited by Jeong-Hwan Moon, IRL, Pusan National UNIV. mwg900@naver.com
""" 

def autonomous_driving(action, done):
    # Reward Policy
    if action == 0:
        reward = 5
    elif (action == 1) or (action == 2):
        reward = -1
    elif (action == 3) or (action == 4):
        reward = -10
    if done:            # 충돌  
        reward = -300

    return reward
    
def autonomous_driving1(action, done):  #2017/7/29/1
    # Reward Policy
    if action == 0:
        reward = 5
    elif (action == 1) or (action == 2):
        reward = -1
    elif (action == 3) or (action == 4):
        reward = -5
    if done:            # 충돌  
        reward = -100

    return reward

# def autonomous_driving1(action, done):  #2017/8/7
#     # Reward Policy
#     if action == 0:
#         reward = 5
#     elif (action == 1) or (action == 2):
#         reward = -1
#     elif (action == 3) or (action == 4):
#         reward = -10
#     if done:            # 충돌  
#         reward = -100
# 
#     return reward
# def autonomous_driving1(action, done):  #2017/7/29/1
#     # Reward Policy
#     if action == 0:
#         reward = 5
#     elif (action == 1) or (action == 2):
#         reward = 0.3
#     elif (action == 3) or (action == 4):
#         reward = -10
#     if done:            # 충돌  
#         reward = -300
# 
#     return reward

#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Environment register for q-learning
Edited by Jeong-Hwan Moon, IRL, Pusan National UNIV. mwg900@naver.com
""" 

class environment :
    #------------------------------------------------------------------------------ 
    #autonomous_driving
    #------------------------------------------------------------------------------ 
    
    #input = 9
    v0 = {          
            'input_size'    :   9,                    # [120, 135, 150, 165, 180, 195, 210, 225, 240] 도의 거리값
            'output_size'   :   5,
            'policy'        :   'autonomous_driving'
         }
    
    #input = 11
    v1 = {          
            'input_size'    :   11,                    # [95, 120, 135, 150, 165, 180, 195, 210, 225, 240, 265] 도의 거리값
            'output_size'   :   5,
            'policy'        :   'autonomous_driving1'
         }
    
    #input = 16 (11 + 5(action_buf))
    v2 = {          
            'input_size'    :   16,                    # [95, 120, 135, 150, 165, 180, 195, 210, 225, 240, 265] 도의 거리값 + 5개 action buffer
            'output_size'   :   5,
            'policy'        :   'autonomous_driving1'
         }
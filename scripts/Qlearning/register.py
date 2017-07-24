#!/usr/bin/env python
#-*- coding: utf-8 -*-

class environment :
    #------------------------------------------------------------------------------ 
    #autonomous_driving
    #------------------------------------------------------------------------------ 
    
    v0 = {          #input = 9
            'input_size'    :   9,                    # [120, 135, 150, 165, 180, 195, 210, 225, 240] 도의 거리값
            'output_size'   :   5,
            'policy'        :   'autonomous_driving'
         }
    
    v1 = {          #input = 11
            'input_size'    :   11,                    # [95, 120, 135, 150, 165, 180, 195, 210, 225, 240, 265] 도의 거리값
            'output_size'   :   5,
            'policy'        :   'autonomous_driving'
         }
    
    
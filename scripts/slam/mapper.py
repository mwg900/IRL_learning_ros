#!/usr/bin/env python
#-*- coding: utf-8 -*-
""" Simple occupancy-grid-based mapping without localization. 

Subscribed topics:
/scan

Published topics:
/map 
/map_metadata

Author: Nathan Sprague
Modify : Jeong-Hwan Moon
    --> Apply OpenCV API to help make occupancy grid

Version: 2/13/14
"""
import rospy
import cv2
import numpy as np
import imutils
import math 
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import LaserScan

import numpy as np


RESOLUTION = 0.05
WIDTH = int(20/RESOLUTION)
HEIGHT = int(20/RESOLUTION)
ORIGIN_X = 10
ORIGIN_Y = 10

class Map(object):
    """ 
    The Map class stores an occupancy grid as a two dimensional
    numpy array. 
    
    Public instance variables:

        width      --  Number of columns in the occupancy grid.
        height     --  Number of rows in the occupancy grid.
        resolution --  Width of each grid square in meters. 
        origin_x   --  Position of the grid cell (0,0) in 
        origin_y   --    in the map coordinate system.
        grid       --  numpy array with height rows and width columns.
        
    
    Note that x increases with increasing column number and y increases
    with increasing row number. 
    """

    def __init__(self, origin_x=ORIGIN_X, origin_y=ORIGIN_Y, resolution=RESOLUTION, 
                 width=WIDTH, height=HEIGHT):
        """ Construct an empty occupancy grid.
        
        Arguments: origin_x, 
                   origin_y  -- The position of grid cell (0,0) in the
                                map coordinate frame.
                   resolution-- width and height of the grid cells 
                                in meters.
                   width, 
                   height    -- The grid will have height rows and width
                                columns cells.  width is the size of
                                the x-dimension and height is the size
                                of the y-dimension.
                                
         The default arguments put (0,0) in the center of the grid. 
                                
        """
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.resolution = resolution
        self.width = width
        self.height = height
        #self.grid = np.zeros((height, width))           #0의 값으로 배열 [h*w] 생성
        self.grid = np.zeros((height,width))
        self.grid_img = np.zeros((height,width))
        self.grid_img.fill(-0.01)
        #print(self.grid_img)
        
    def to_message(self, img):
        """ Return a nav_msgs/OccupancyGrid representation of this map. """
     
        grid_msg = OccupancyGrid()

        # Set up the header.
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "map"

        # .info is a nav_msgs/MapMetaData message. 
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.width
        grid_msg.info.height = self.height
        
        # Rotated maps are not supported... quaternion represents no
        # rotation. 
        grid_msg.info.origin = Pose(Point(self.origin_x, self.origin_y, 0),
                               Quaternion(0, 0, 0, 1))
        
        # Flatten the numpy array into a list of integers from 0-100.
        # This assumes that the grid entries are probalities in the
        # range 0-1. This code will need to be modified if the grid
        # entries are given a different interpretation (like
        # log-odds).
        
        #print(self.grid.size)
        
        #flat_grid = self.grid.reshape((self.grid.size,)) * 100      #100을 곱해줌으로써 0~1의 값을 0~100으로 만들어줌. 이는 msg형식에서 필수
        flat_grid = img.reshape((img.size,))*100
        #print(flat_grid)
        grid_msg.data = list(np.round(flat_grid))
        return grid_msg

    def set_cell(self, x, y, val):
        """ Set the value of a cell in the grid. 

        Arguments: 
            x, y  - This is a point in the map coordinate frame.
            val   - This is the value that should be assigned to the
                    grid cell that contains (x,y).

        This would probably be a helpful method!  Feel free to throw out
        point that land outside of the grid. 
        """
        pass

class Mapper(object):
    """ 
    The Mapper class creates a map from laser scan data.
    """
    
    def __init__(self):
        """ Start the mapper. """
        
        rospy.init_node('mapper')
        self._map = Map()
        self.F = False
        self.rate = rospy.Rate(10) # 5hz
        # Setting the queue_size to 1 will prevent the subscriber from
        # buffering scan messages.  This is important because the
        # callback is likely to be too slow to keep up with the scan
        # messages. If we buffer those messages we will fall behind
        # and end up processing really old scans.  Better to just drop
        # old scans and always work with the most recent available.
        rospy.Subscriber('scan',
                         LaserScan, self.scan_callback, queue_size=1)

        # Latched publishers are used for slow changing topics like
        # maps.  Data will sit on the topic until someone reads it. 
        self._map_pub = rospy.Publisher('map', OccupancyGrid, latch=True, queue_size = 100)
        self._map_data_pub = rospy.Publisher('map_metadata', MapMetaData, latch=True, queue_size = 100)
        self.map_img = self._map.grid_img
        print("Mapper ready")
        #rospy.spin()
        
    def meter_to_pixel(self, dist):
        pixel = dist / RESOLUTION
        return pixel

    def scan_callback(self, LaserScan):
        """ Update the map on every scan callback. """
        
        # Fill some cells in the map just so we can see that something is 
        # being published. 
        '''
        self._map.grid[0, 0] = 1.0
        self._map.grid[0, 1] = .9
        self._map.grid[0, 2] = .7
        self._map.grid[1, 0] = .5
        self._map.grid[2, 0] = .3
        '''
        
        # Now that the map is updated, publish it!
        
        
        
        for ang in range(len(LaserScan.ranges)):
            center_x = int(ORIGIN_X/RESOLUTION)
            center_y = int(ORIGIN_Y/RESOLUTION)
            if (LaserScan.ranges[ang] != 0) and (LaserScan.ranges[ang] != 6.0):
                dist = self.meter_to_pixel(LaserScan.ranges[ang])
                
                x = int(center_x+dist*math.cos(math.radians(ang)))
                y = int(center_y+dist*math.sin(math.radians(ang)))
                cv2.line(self.map_img, (center_x,center_y), (x,y), (0))   #라이닝
                self.map_img[y, x] = 1.0
        cv2.imshow("grid_img",self.map_img)
        #cv2.waitKey(0)
        #rospy.loginfo("Scan is processed, publishing updated map.")
        self.F = True
        

    def publish_map(self):
        """ Publish the map. """
        grid_msg = self._map.to_message(self.map_img)
        #print(grid_msg)
        self._map_data_pub.publish(grid_msg.info)
        self._map_pub.publish(grid_msg)
        
    def main(self):
        while not rospy.is_shutdown():
            if self.F is True:
                #cv2.imshow("grid_img",self._map.grid_img)
                self.publish_map()
                self.rate.sleep()
                
if __name__ == '__main__':
    try:
        m = Mapper()
        m.main()
    except rospy.ROSInterruptException:
        pass
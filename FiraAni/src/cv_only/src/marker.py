#!/usr/bin/env python3

from cv_only.msg import LaneCoordinates
import numpy as np
import rospy
import matplotlib.pyplot as plt
from std_msgs.msg import Bool,Float64MultiArray, Int64MultiArray, Float64
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseArray

def callback(msg):
    marker = Marker()
    points = []
    for pose in msg.poses:
        points.append(pose.position)
    marker.points = points
    marker.header.frame_id = 'camera_color_optical_frame'
    marker.pose.position.x = 1.0
    marker.pose.position.y = 1.0
    marker.pose.position.z = 0.0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0 
    marker.pose.orientation.z = 0.707
    marker.pose.orientation.w = 0.707           
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05
    pub.publish(marker)    

rospy.init_node("mp_data")

pub=rospy.Publisher('/path_markers',Marker,queue_size=10)        
rospy.Subscriber('/best_trajectory', PoseArray, callback)
rospy.spin()


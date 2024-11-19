#! /usr/bin/python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid

f = open('grid18.txt','r')
s = f.readlines()
data = [int(i.strip()) for i in s[3].lstrip('(').rstrip(')\n').split(", ")]
pose = s[1][12:].split(",")
x=float(pose[0].strip())
y=float(pose[1][5:].strip())

grid = OccupancyGrid()
grid.header.frame_id = "camera_color_optical_frame"
grid.info.resolution = 0.008
grid.info.width = 250
grid.info.height = 250
grid.info.origin.position.x = 0
grid.info.origin.position.y = 0
grid.info.origin.position.z = 0
grid.info.origin.orientation.x = 0
grid.info.origin.orientation.y = 0
grid.info.origin.orientation.z = 0
grid.info.origin.orientation.w = 1
grid.data = data

rospy.init_node("dummy_goal")
r = rospy.Rate(20)
goal_pub = rospy.Publisher('/local_goal_dm',PoseStamped,queue_size=10)
grid_pub = rospy.Publisher('/occupancy_grid',OccupancyGrid,queue_size=10)
goal = PoseStamped()
goal.header.frame_id = "camera_color_optical_frame"
goal.pose.position.x = x
goal.pose.position.y = y
goal.pose.orientation.x = 0
goal.pose.orientation.y = 0
goal.pose.orientation.z = 0
goal.pose.orientation.w = 1
while not rospy.is_shutdown():
    goal_pub.publish(goal)
    grid_pub.publish(grid)
    r.sleep()

#! /usr/bin/python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Float64
import torch 


rospy.init_node("dummy_path")
r = rospy.Rate(10)
path_pub = rospy.Publisher('/best_trajectory',PoseArray,queue_size=10)
traj = PoseArray()
x = np.linspace(0,1,100)
for i in range(1,100):
    traj1 = Pose()
    traj1.position = Point(x[i],np.cos(-x[i])-1,0.3)
    traj1.orientation = Quaternion(0,0,0,0)
    traj.poses.append(traj1)

path_pub.publish(traj)
while not rospy.is_shutdown():
    path_pub.publish(traj)
    r.sleep()


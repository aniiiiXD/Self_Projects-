#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64

def cbk():
    

def main():
    global pub
    rospy.init_node("velocity_tester_node")
    pub = rospy.Publisher("/velocity", Float64, cbk)



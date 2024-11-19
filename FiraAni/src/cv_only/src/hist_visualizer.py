#!/usr/bin/env python3

import rospy
import matplotlib.pyplot as plt
from std_msgs.msg import Float64MultiArray
import numpy as np


def hist_cbk(msg):
    global smooth_histogram
    smooth_histogram = []
    for i in msg.data:
        smooth_histogram.append(i)


def max_cbk(msg):
    global smooth_histogram
    maximas = []
    for i in msg.data:
        maximas.append(i)
        print(i)

    numbers = np.arange(640) 
    plt.figure(figsize=(10, 10))
    plt.plot(numbers, smooth_histogram)
    plt.scatter(maximas, np.zeros(len(maximas)))
    plt.show()


rospy.init_node("hist_node")
rospy.Subscriber("/histogram", Float64MultiArray, hist_cbk)
rospy.Subscriber("/maximas", Float64MultiArray, max_cbk)
rospy.spin()

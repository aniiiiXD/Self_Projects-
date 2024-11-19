#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import time


class dummy_node:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            '/lane_img', Image, self.image_callback)
        self.image_pub = rospy.Subscriber(
            '/erosion', Image, self.erosion_cbk)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        cv2.imwrite("lane.png", cv_image)

    def erosion_cbk(self, msg):
        erosion = self.bridge.imgmsg_to_cv2(msg, "mono8")

        cv2.imwrite("erosion.png", erosion)


if __name__ == '__main__':
    rospy.init_node('dummy_node')
    node = dummy_node()
    rospy.spin()

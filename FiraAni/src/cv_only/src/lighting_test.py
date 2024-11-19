#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import time

class LaneDetectionNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            '/camera/color/image_raw', Image, self.image_callback)
        self.image_pub = rospy.Publisher(
            '/lighting_test', Image, queue_size=1)

    def image_callback(self, msg):
        start = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        blur_size = 5
        blurred_image = cv2.GaussianBlur(gray_image, (blur_size, blur_size), 0)

        equalized_image = cv2.equalizeHist(blurred_image)

        # gamma = 0.7
        # table = [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
        # table = np.array(table, np.uint8)
        # gamma_img = cv2.LUT(equalized_image, table)

        thresh, edges = cv2.threshold(equalized_image, 190, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        output_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")

        self.image_pub.publish(output_msg)
        end = time.time()

        print(end - start)

if __name__ == '__main__':
    rospy.init_node('lane_detection_node')
    node = LaneDetectionNode()
    rospy.spin()

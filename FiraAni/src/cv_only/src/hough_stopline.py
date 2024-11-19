#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool

class StopLineRemover:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('stop_line_remover')

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber('/erosion', Image, self.image_callback)
        self.img_pub = rospy.Publisher('/stopline_img', Image, queue_size=1)
        self.stopline_pub = rospy.Publisher('/stopline_flag', Bool, queue_size=1)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        # cv_image = cv2.imread("/home/yuvi/Downloads/IMG-20240807-WA0001.jpg")
        # gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(cv_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 50, 150, apertureSize=3)

        # Apply dilation to strengthen the edges
        kernel = np.ones((2, 2), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        # Detect lines using HoughLinesP
        rho = 1  # Distance resolution in pixels
        theta = np.pi / 180  # Angle resolution in radians (1 degree)
        threshold = 100  # Minimum number of votes (intersections in Hough grid cell)
        minLineLength = 250  # Minimum number of pixels making up a line
        maxLineGap = 30  # Maximum gap in pixels between connectable line segments

        lines = cv2.HoughLinesP(dilated_edges, rho, theta, threshold, minLineLength, maxLineGap)

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if abs(y2 - y1) < 5: # Angle
                        height = 10
                        if y1 < y2:
                            top_left = (x1, y1 - height // 2)
                            bottom_right = (x2, y2 + height // 2)
                        else:
                            top_left = (x1, y1 + height // 2)
                            bottom_right = (x2, y2 - height // 2)
                        
                        # Draw the rectangle to blacken out the line
                        cv2.rectangle(cv_image, top_left, bottom_right, (0, 0, 0), -1)
        # Display the result
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image))
        self.stopline_pub.publish(True)
        rospy.loginfo("stopline!!!!!!!!!1")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    stop_line_remover = StopLineRemover()
    stop_line_remover.run()

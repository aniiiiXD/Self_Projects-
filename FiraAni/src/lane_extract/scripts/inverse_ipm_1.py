#! /usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class IPM():
    def __init__(self):
        rospy.init_node('inverse_ipm_node')
        self.lanes_sub = rospy.Subscriber("/lane_image", Image, self.callback)
        self.ipm_lanes_pub = rospy.Publisher("/inverse_lanes_topic", Float64MultiArray, queue_size=10)
        self.bridge = CvBridge()
    
    def callback(self, msg):
        # Convert ROS Image message to OpenCV image (RGB)
        rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Resize the image to half of its original height
        rgb_img_half = cv2.resize(rgb_img, (640, 240), interpolation=cv2.INTER_AREA)
        
        # Create an empty canvas of the original size
        original = np.zeros((480, 640, 3), np.uint8)
        height = 480
        
        # Place the resized image in the bottom half of the canvas
        original[height//2:, :] = rgb_img_half
        
        # Define perspective transformation parameters
        (w, h) = (640, 480)
        ymax = 480
        x1 = w // 2 - 300
        x2 = w // 2 + 175
        l = 400
        tl = (180, 320)
        bl = (10, 460)
        tr = (400, 320)
        br = (500, 460)

        source = np.array([bl, br, tr, tl], dtype="float32")
        destination = np.float32([[x1, ymax], [x2, ymax], [x2, ymax-l], [x1, ymax-l]])

        # Compute the inverse perspective transform matrix
        M_inv = cv2.getPerspectiveTransform(destination, source)

        # Apply the inverse perspective transform
        warped = cv2.warpPerspective(original, M_inv, (640, 480), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Resize the warped image for visualization or further processing
        original_cam = cv2.resize(warped, (1280, 720), interpolation=cv2.INTER_LINEAR)
        
        # Extract points where any of the RGB channels have non-zero values
        b, g, r = cv2.split(original_cam)
        points = np.where(np.logical_or(b > 0, np.logical_or(r > 0, g > 0)))
        points = np.column_stack((points[1], points[0]))

        # Convert the points array to a Float64MultiArray message
        points_msg = Float64MultiArray()
        points_msg.data = points.flatten()

        # Publish the points
        self.ipm_lanes_pub.publish(points_msg)

if __name__ == "__main__":
    obj = IPM()
    rospy.spin()

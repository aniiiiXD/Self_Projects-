#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
from geometry_msgs.msg import PoseStamped


class SlidingWindow():

    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('SlidingWindow')
        self.set_pub_sub()
        self.set_ipm_params()

        self.local_goal = PoseStamped()

    def set_pub_sub(self):
        rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        self.image_pub = rospy.Publisher('/lane_image', Image, queue_size=10)
        self.local_goal_pub = rospy.Publisher(
            "/local_goal_dm", PoseStamped, queue_size=1)

    def set_ipm_params(self):
        self.h, self.w = 480, 640
        phy_length = 60
        self.res_x = 1 / (100 * 0.00140625)
        self.res_y = 1 / (100 * 0.0010625)
        pxls_per_cm = self.h / phy_length
        self.search_coord = self.h // 2

        tl = (rospy.get_param("/TL_X", 120), rospy.get_param("/TL_Y", 350))
        bl = (rospy.get_param("/BL_X", -110), rospy.get_param("/B_Y", 480))
        tr = (rospy.get_param("/TR_X", 350), rospy.get_param("/TL_Y", 350))
        br = (rospy.get_param("/BR_X", 390), rospy.get_param("/B_Y", 480))

        src_pts = [bl, br, tr, tl]
        src_pts = np.array(src_pts, dtype='float32')

        dest_pts = np.array([
            (self.w / 2 - 40 * self.res_x, self.h),
            (self.w / 2 + 20 * self.res_x, self.h),
            (self.w / 2 + 20 * self.res_x, 0),
            (self.w / 2 - 40 * self.res_x, 0)
        ], dtype='float32')

        self.M = cv2.getPerspectiveTransform(src_pts, dest_pts)

    def get_red_img(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask = cv2.bitwise_or(mask1, mask2)
        red_img = cv2.bitwise_and(img, img, mask=mask)

        gray_image = cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY)

        _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

        return binary_image

    def find_goal_x_pxl(self, img):
        middle_line_y = self.search_coord
        white_pixels_x = np.where(img[middle_line_y] == 255)[0]

        if len(white_pixels_x) > 0:
            average_x = np.mean(white_pixels_x)
            return average_x
        else:
            return None

    def find_real_coords(self, x):
        y = (- self.search_coord + self.h) * 0.0010625
        x = (x - 415) * 0.00140625

        return (x, y)

    def publish_goal(self, coords):
        self.local_goal.header.stamp = rospy.Time.now()
        self.local_goal.header.frame_id = 'camera_color_optical_frame'
        self.local_goal.pose.position.y = coords[1]
        self.local_goal.pose.position.x = coords[0]
        self.local_goal.pose.position.z = 0
        self.local_goal.pose.orientation.x = 0
        self.local_goal.pose.orientation.y = 0
        self.local_goal.pose.orientation.z = 0
        self.local_goal.pose.orientation.w = 1
        self.local_goal_pub.publish(self.local_goal)

    def callback(self, msg):
        start_time = time.time()
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # img = cv2.imread('/home/yuvi/Downloads/IMG-20240809-WA0017.jpg')
        resized_img = cv2.resize(
            img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        ipm_img = cv2.warpPerspective(
            resized_img, self.M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        red_img = self.get_red_img(ipm_img)

        goal_x_pxl = self.find_goal_x_pxl(red_img)
        if goal_x_pxl == None:
            print("NO RED COORDINATES FOUND")
            end_time = time.time()
            print(end_time-start_time)
            return

        print("GOAL IMAGE COORDINATES: ", goal_x_pxl, self.search_coord)

        real_coords = self.find_real_coords(goal_x_pxl)

        self.publish_goal(real_coords)
        self.image_pub.publish(red_img)

        end_time = time.time()
        print(end_time-start_time)


if __name__ == '__main__':
    SlidingWindow()
    rospy.spin()

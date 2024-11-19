#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import time
from scipy.signal import argrelextrema
import scipy.ndimage


class LaneDetection:
    def __init__(self):
        self.bridge = CvBridge()
        self.gamma = rospy.get_param('~gamma', 0.5)

        self.image_sub = rospy.Subscriber("/camera/color/image_raw",
                                          Image, self.image_callback)
        self.image_pub = rospy.Publisher("/erosion", Image, queue_size=1)
        self.box_pub = rospy.Publisher("/box", Image, queue_size=1)

        self.init_bool = False
        # self.box_img = np.zeros((480, 640, 3), dtype=np.uint8)  # Initialize with zeros and the same shape as your images

    def set_img_params(self):
        phy_length = 75
        self.h = 480
        self.w = 640
        cm_to_pxl = self.h // phy_length
        x1 = int(self.w // 2 - 40 * cm_to_pxl)
        x2 = int(self.w // 2 + 20 * cm_to_pxl)
        tl = (215, 345)
        bl = (50, 465)
        tr = (410, 345)
        br = (565, 465)
        source = np.array([bl, br, tr, tl], dtype="float32")
        destination = np.float32(
            [[x1, self.h], [x2, self.h], [x2, 0], [x1, 0]])
        self.M = cv2.getPerspectiveTransform(source, destination)

    def set_win_params(self):
        self.nwin = 15
        self.minpix = 300
        self.margin = 50
        self.win_ht = self.h // self.nwin

    def process_img(self, img):
        # img = cv2.warpPerspective(img, self.M, (self.w, self.h), flags=cv2.INTER_LINEAR,
        #                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        gamma = 0.5
        table = [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        gamma_img = cv2.LUT(img, table)

        gray_img = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2GRAY)
        binary_threshold = 130
        thresh, edges = cv2.threshold(gray_img, binary_threshold, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(edges, kernel, iterations=6)
        # canny = cv2.Canny(erosion, 50, 150)
        return erosion

    def get_left_lane(self, x, y):
        left_ind = np.array([])
        left_curr = self.left_base
        ctr = 0
        for win_num in range(self.nwin):
            y_low = self.h - (win_num + 1) * self.win_ht
            y_high = self.h - win_num * self.win_ht
            x_low = left_curr - self.margin
            x_high = left_curr + self.margin
            win_ind = ((y >= y_low) & (y < y_high) & (x >= x_low)
                       & (x < x_high)).nonzero()[0]

            if len(win_ind) >= self.minpix:
                ctr += 1
                img_ind = x[win_ind]
                left_ind = np.concatenate((left_ind, img_ind))
                left_curr = np.int64(np.mean(img_ind))
                cv2.rectangle(self.box_img, (x_low, y_low),
                              (x_high, y_high), (255, 0, 0), 4)
                if ctr == 1:
                    self.left_base = left_curr
            elif ctr > 0:  # If some ind have already been added
                break
        return left_ind

    def get_right_lane(self, x, y):
        right_ind = np.array([])
        right_curr = self.right_base
        ctr = 0
        for win_num in range(self.nwin):
            y_low = self.h - (win_num + 1) * self.win_ht
            y_high = self.h - win_num * self.win_ht
            x_low = right_curr - self.margin
            x_high = right_curr + self.margin
            win_ind = ((y >= y_low) & (y < y_high) & (x >= x_low)
                       & (x < x_high)).nonzero()[0]

            if len(win_ind) >= self.minpix:
                ctr += 1
                img_ind = x[win_ind]
                right_ind = np.concatenate((right_ind, img_ind))
                right_curr = np.int64(np.mean(img_ind))
                cv2.rectangle(self.box_img, (x_low, y_low),
                              (x_high, y_high), (0, 0, 255), 4)

                if ctr == 1:
                    self.right_base = right_curr
            elif ctr > 0:  # If some ind have already been added
                break
        return right_ind

    def get_color_img(self, img):
        x = np.array(img.nonzero()[0])
        y = np.array(img.nonzero()[1])
        left_ind, right_ind = np.array([]), np.array([])

        if self.left_base >= 0 and self.right_base <= self.w:
            left_ind = self.get_left_lane(x, y).astype(int)
            right_ind = self.get_right_lane(x, y).astype(int)
        else:
            print("Not completed this case")

        colored_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        if len(left_ind) > 0:
            colored_img[y[left_ind],
                        x[left_ind]] = [255, 0, 0]
        if len(right_ind) > 0:
            colored_img[y[right_ind],
                        x[right_ind]] = [0, 0, 255]

        return colored_img

    def image_callback(self, msg):
        start_time = time.time()

        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        erosion = self.process_img(img)
        self.box_img = cv2.cvtColor(erosion, cv2.COLOR_GRAY2BGR)

        if not self.init_bool:
            histogram = np.sum(erosion / 255, axis=0)
            smooth_histogram = scipy.ndimage.gaussian_filter(histogram,
                                                             sigma=30)
            maxima_indices = argrelextrema(smooth_histogram, np.greater)[0]
            self.left_base = maxima_indices[0]
            self.right_base = maxima_indices[-1]
            self.init_bool = True

        else:
            # colored_img = self.get_color_img(erosion)
            processed_image_msg = self.bridge.cv2_to_imgmsg(erosion,
                                                            "passthrough")
            self.image_pub.publish(processed_image_msg)
            # self.box_pub.publish(self.bridge.cv2_to_imgmsg(self.box_img,
            #                                                "passthrough"))

        end_time = time.time()
        tot_time = end_time - start_time
        print(tot_time)


if __name__ == "__main__":
    rospy.init_node('SlidingWindowNode')
    node = LaneDetection()
    node.set_img_params()
    node.set_win_params()
    rospy.spin()

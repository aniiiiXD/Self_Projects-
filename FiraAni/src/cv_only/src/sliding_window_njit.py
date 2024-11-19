#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from sensor_msgs.msg import Image
from scipy.signal import argrelextrema
from std_msgs.msg import Bool,Float64MultiArray, Int64MultiArray, Float64
from cv_bridge import CvBridge
import time
from cv_only.msg import LaneCoordinates
from threading import *
from numba import njit


@njit
def avg_lane(img):
    out_img = np.zeros_like(img)
    y_indices, x_indices = np.nonzero(img)[0], np.nonzero(img)[1]

    for y in np.unique(y_indices):
        x_index = x_indices[y_indices == y]
        avg_x_index = int(np.mean(x_index))
        out_img[y, avg_x_index] = 255

    return out_img


def process_img(img):
    image = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    ymax = 480
    
    #  Racing
    # x1 = w // 2 - 38 * 8
    # x2 = w // 2 + 14 * 8

    x1 = w // 2 - 40 * 8
    x2 = w // 2 + 20 * 8

    l = 60 * 8
    tl = (100, 350)
    bl = (2, 433)
    tr = (350, 350)
    br = (417, 433)

    # cv2.circle(image, tl, 5, (0, 0, 255), -1)
    # cv2.circle(image, bl, 5, (0, 0, 255), -1)
    # cv2.circle(image, tr, 5, (0, 0, 255), -1)
    # cv2.circle(image, br, 5, (0, 0, 255), -1)

    # cv2.imshow("Original", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    source = np.array([bl, br, tr, tl], dtype="float32")
    destination = np.float32(
        [[x1, ymax], [x2, ymax], [x2, ymax - l], [x1, ymax - l]])
    M = cv2.getPerspectiveTransform(source, destination)
    image = cv2.warpPerspective(
        image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    input_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_threshold = rospy.get_param('BINARY_THRESHOLD')

    ret, edges = cv2.threshold(input_image_gray, binary_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)

    erosion = cv2.erode(edges, kernel, iterations=6)
    erosion_half = erosion[erosion.shape[0] // 2:, :]
    erosion_half_resize = cv2.resize(
        erosion_half, (640, 480), interpolation=cv2.INTER_LINEAR)
    erosion_half_resize[:, 0:30] = 0
    erosion_half_resize[:, -30:] = 0
    return erosion_half_resize


@njit
def get_histogram(image):
    img_ratio = rospy.get_param('IMG_RATIO')
    histogram = np.sum(
        image[int(image.shape[0] * (1 - img_ratio)):, :] / 255, axis=0)

    return histogram


@njit
def concatenation(arr):
    final_arr = []
    for i in arr:
        for j in i:
            final_arr.append(j)
    return np.array(final_arr)


@njit
def find_lane_pixels(image, smooth_histogram):
    midpoint = np.int64(smooth_histogram.shape[0] // 2)

    # numbers = np.arange(image.shape[1])  # Use np.arange to match the shape
    # plt.figure(figsize=(10, 10))
    # plt.plot(numbers, smooth_histogram)
    # Draw a vertical line at x = 320
    # plt.axvline(x=midpoint, color='r', linestyle='--')
    # plt.show()

    out_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # cv2.imshow('Windows', out_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    leftx_base = np.argmax(smooth_histogram[:midpoint])
    rightx_base = np.argmax(smooth_histogram[midpoint:]) + midpoint

    nwindows = rospy.get_param('NWINDOWS')
    minpix = rospy.get_param('MINPIX')
    margin = rospy.get_param('MARGIN')

    window_height = np.int64(image.shape[0] // nwindows)

    nonzero = image.nonzero()
    nonzeroy = (nonzero[0])
    nonzerox = (nonzero[1])

    leftx_current = leftx_base
    left_lane_inds = []
    hit_left = False

    for window in range(nwindows):
        win_y_low = image.shape[0] - (window + 1) * window_height
        win_y_high = image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        global img_color

        if len(good_left_inds) < minpix:
            if hit_left:
                break
            else:
                cv2.rectangle(img_color, (win_xleft_low, win_y_low),
                            (win_xleft_high, win_y_high), (255, 0, 0), 4)
                continue

        else:
            hit_left = True
            leftx_current = np.int64(1 * np.mean(nonzerox[good_left_inds]))
            cv2.rectangle(img_color, (win_xleft_low, win_y_low),
                            (win_xleft_high, win_y_high), (255, 0, 0), 4)
            left_lane_inds.append(good_left_inds)

    rightx_current = rightx_base
    right_lane_inds = []
    hit_right = False

    for window in range(nwindows):
        win_y_low = image.shape[0] - (window + 1) * window_height
        win_y_high = image.shape[0] - window * window_height
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        if len(good_right_inds) < minpix:
            if hit_right:
                break
            else:
                continue
        right_lane_inds.append(good_right_inds)

        if len(good_right_inds) > minpix:
            rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))
            hit_right = True

    left_lane_inds = (concatenation(left_lane_inds))
    right_lane_inds = (concatenation(right_lane_inds))

    # left_lane_inds = np.concatenate(
    #     left_lane_inds) if left_lane_inds else np.array([])
    # right_lane_inds = np.concatenate(
    #     right_lane_inds) if right_lane_inds else np.array([])

    # leftx = nonzerox[left_lane_inds] if len(left_lane_inds) else np.array([
    # ])
    # lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) else np.array([
    # ])
    # rightx = nonzerox[right_lane_inds] if right_lane_inds.size else np.array([
    # ])
    # righty = nonzeroy[right_lane_inds] if right_lane_inds.size else np.array([
    # ])

    # PRONE TO ERRORS, CHECK THE ELSE STATEMENTS!!!!!!!
    if len(left_lane_inds):
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
    if len(right_lane_inds):
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def search_around_poly(image):

    nonzero = image.nonzero()
    nonzeroy = (nonzero[0])
    nonzerox = (nonzero[1])
    histogram = get_histogram(image)
    smooth_histogram = scipy.ndimage.gaussian_filter(histogram, sigma=30)

    margin = rospy.get_param('MARGIN')  
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(
        image, smooth_histogram)
    len_left, len_right = len(leftx), len(rightx)

    if len_left:
        left_fit = np.polyfit(lefty, leftx, 2)
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                           2] + margin)))
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        out_img[nonzeroy[left_lane_inds],
                nonzerox[left_lane_inds]] = [255, 0, 0]

    if len_right:
        right_fit = np.polyfit(righty, rightx, 2)
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
                                            2] + margin)))
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        out_img[nonzeroy[right_lane_inds],
                nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img


def get_lane_dist(b, r):
    valid_dist = []

    for y in range(b.shape[0] // 2, b.shape[0]):
        b_x = np.nonzero(b[y])[0]
        r_x = np.nonzero(r[y])[0]

        if len(b_x) == 1 and len(r_x) == 1:
            dist = abs(b_x[0] - r_x[0])
            valid_dist.append(dist)

    if valid_dist:
        avg_dist = np.mean(valid_dist)
    else:
        avg_dist = None

    return avg_dist


def form_middle_lane(b, r):
    img = cv2.merge([b, np.zeros_like(b), r])
    for y in range(img.shape[0]):
        x_coords = np.nonzero(img[y] == 255)[0]

        if len(x_coords) >= 2:
            # Get the first two x-coordinates where img[y] == 255
            x1, x2 = x_coords[:2]
            midpoint = (x1 + x2) // 2 

            img[y, midpoint] = [0, 255, 0]
    return img


def form_right_or_left_lane(b, r):
    y_b = np.nonzero(b)[0] 
    y_r = np.nonzero(r)[0]  

    if len(y_r) >= len(y_b):
        img = cv2.merge([np.zeros_like(b), b, r])

        for y in range(b.shape[0]):
            x_coords_b = np.nonzero(b[y] == 255)[0]
            x_coords_r = np.nonzero(r[y] == 255)[0]

            if len(x_coords_b) >= 1 and len(x_coords_r) >= 1:
                x1 = x_coords_b[0]
                x2 = x_coords_r[0]
                new_x = 2 * x1 - x2
                if 0 <= new_x < img.shape[1]:
                    img[y, new_x] = (255, 0, 0)
    else:
        img = cv2.merge([b, r, np.zeros_like(b)])

        for y in range(r.shape[0]):
            x_coords_b = np.nonzero(b[y] == 255)[0]
            x_coords_r = np.nonzero(r[y] == 255)[0]

            if len(x_coords_b) >= 1 and len(x_coords_r) >= 1:
                x1 = x_coords_b[0]
                x2 = x_coords_r[0]
                new_x = 2 * x2 - x1
                if 0 <= new_x < img.shape[1]:
                    img[y, new_x] = (0, 0, 255)

    return img


def form_single_lane(b, r):
    if b.any():
        y_indices, x_indices = np.nonzero(b)[0], np.nonzero(b)[1]
        min_x = np.min(x_indices)
        max_x = np.max(x_indices)
        min_x_index = np.where(x_indices == min_x)[0][0]
        max_x_index = np.where(x_indices == max_x)[0][0]

        y1 = 480 - y_indices[min_x_index]
        y2 = 480 - y_indices[max_x_index]

        slope = (y2 - y1) / (max_x - min_x)

        if slope >= 0:
            img = cv2.merge([b, np.zeros_like(b), np.zeros_like(b)])
        else:
            img = cv2.merge([np.zeros_like(b), np.zeros_like(b), b])

    else:
        y_indices, x_indices = np.nonzero(r)[0], np.nonzero(r)[1]
        min_x = np.min(x_indices)
        max_x = np.max(x_indices)
        min_x_index = np.where(x_indices == min_x)[0][0]
        max_x_index = np.where(x_indices == max_x)[0][0]

        y1 = 480 - y_indices[min_x_index]
        y2 = 480 - y_indices[max_x_index]

        slope = (y2 - y1) / (max_x - min_x)

        if slope >= 0:
            img = cv2.merge([r, np.zeros_like(r), np.zeros_like(r)])
        else:
            img = cv2.merge([np.zeros_like(r), np.zeros_like(r), r])

    return img


def visualize_lane_detection(image):
    out_img = search_around_poly(image)

    b, g, r = cv2.split(out_img)
    b = avg_lane(b)
    r = avg_lane(r)

    if b.any() and r.any():
        dist = get_lane_dist(b, r)
        threshold_dist = rospy.get_param('THRESHOLD_DIST')

        if dist > threshold_dist:
            final = form_middle_lane(b, r)
        else:
            final = form_right_or_left_lane(b, r)

    else:
        final = form_single_lane(b, r)

    return final


class SlidingWindow(Thread):

    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        rospy.init_node('SlidingWindow')
        rospy.Subscriber('/middle_lane_bool', Bool, self.middle_callback)
        rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        rospy.Subscriber('/odom', Float64, self.odom_callback)
        self.speed = 0
        self.pub = rospy.Publisher('/lane_image', Image, queue_size=10)
        self.erosion_pub = rospy.Publisher(
            '/erosion_lane_image', Image, queue_size=10)
        self.lane_coordinates_pub = rospy.Publisher(
            "/lane_coordinates", LaneCoordinates, queue_size=10)
        self.stopline_pub = rospy.Publisher(
            '/stopline_topic', Float64, queue_size=10)
        self.window_pub = rospy.Publisher('/window_topic', Image, queue_size=10)
        self.occupancy_points_pub=rospy.Publisher('/occupancy_points',Float64MultiArray,queue_size=10)
        self.img_color = 0
        self.middle_bool = False

    def odom_callback(self, msg):
        self.speed = msg.data

    def remove_stopline(self, erosion):
        stopline = Int64MultiArray()
        histogram_stop = np.sum(erosion / 255, axis=1)
        stopline_threshold = rospy.get_param('STOPLINE_THRESHOLD') 
        max_index = 0
        if np.all(histogram_stop < stopline_threshold):
            stopline.data = [320, max_index]
        else:
            max_index = np.argmax(histogram_stop)
            stopline.data = [320, max_index]
            for i in range(480):
                if histogram_stop[i] > stopline_threshold:
                    erosion[i, :] = 0

        return erosion, max_index

    def calculate_time(self, stopline_y, speed):
        blindspot_dist = 20
        dist = (480 - stopline_y) / 8 + blindspot_dist
        return dist / (100 * speed)

    def callback(self, msg):
        start_time = time.time()
        image = self.bridge.imgmsg_to_cv2(msg)
        erosion = process_img(image)
        stopline_removed_img, stopline_y = self.remove_stopline(erosion)

        if stopline_y == 400:
            speed = self.speed
            remaining_time = Float64()
            remaining_time.data = self.calculate_time(stopline_y, speed)
            self.stopline_pub.publish(remaining_time)

        self.erosion_pub.publish(
            self.bridge.cv2_to_imgmsg(stopline_removed_img))
        final_img = visualize_lane_detection(stopline_removed_img)
        # self.erosion_pub.publish(self.bridge.cv2_to_imgmsg(erosion))
        # final_img = visualize_lane_detection(erosion)

        if self.middle_bool:
            blue, green, red = cv2.split(final_img)
            corrected_img = cv2.merge([blue, np.zeros_like(blue), red])
            corrected_msg = self.bridge.cv2_to_imgmsg(corrected_img)
            self.pub.publish(corrected_msg)
            self.publish_lane_coords(corrected_img)

        else:
            final_msg = self.bridge.cv2_to_imgmsg(final_img)
            self.pub.publish(final_msg)
            self.publish_lane_coords(final_img)

        end_time = time.time()
        print(end_time-start_time)

    def middle_callback(self, msg):
        result = msg.data
        self.middle_bool = result

    def publish_lane_coords(self, image):
        b, g, r = cv2.split(image)
        b_y, b_x = np.nonzero(b)[0], np.nonzero(b)[1]
        g_y, g_x = np.nonzero(g)[0], np.nonzero(g)[1]
        r_y, r_x = np.nonzero(r)[0], np.nonzero(r)[1]

        lane_coordinates = LaneCoordinates()
        lane_coordinates.rx = r_x
        lane_coordinates.ry = r_y
        lane_coordinates.mx = g_x
        lane_coordinates.my = g_y
        lane_coordinates.lx = b_x
        lane_coordinates.ly = b_y

        self.lane_coordinates_pub.publish(lane_coordinates)


if __name__ == '__main__':
    SlidingWindow()
    rospy.spin()

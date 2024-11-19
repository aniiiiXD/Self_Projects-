#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import argrelextrema
from std_msgs.msg import Bool,Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
from cv_only.msg import LaneCoordinates


class SlidingWindow():

    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('SlidingWindow')
        rospy.Subscriber('/middle_lane', Bool, self.middle_callback)
        rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        self.pub = rospy.Publisher('/lane_image', Image, queue_size=10)
        self.occupancy_points_pub=rospy.Publisher('/occupancy_points',Float64MultiArray,queue_size=10)
        self.erosion_pub = rospy.Publisher(
            '/erosion_lane_image', Image, queue_size=10)
        self.lane_coordinates_pub = rospy.Publisher(
            "/lane_coordinates", LaneCoordinates, queue_size=10)
        self.window_pub = rospy.Publisher('/window_topic', Image, queue_size=10)
        self.middle_bool = False
        self.img_color = 0
        self.margin = rospy.get_param('MARGIN')


    def callback(self, msg):
        start_time = time.time()
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
        erosion = self.process_img(image)
        self.img_color = cv2.merge([erosion, erosion, erosion])
        self.erosion_pub.publish(self.bridge.cv2_to_imgmsg(erosion))
        final_img = self.visualize_lane_detection(erosion)

        # print(self.middle_bool)
        if self.middle_bool:
            blue, green, red = cv2.split(final_img)
            corrected_img = cv2.merge([blue, np.zeros_like(blue), red])
            corrected_msg = self.bridge.cv2_to_imgmsg(corrected_img)
            self.pub.publish(corrected_msg)
            # self.publish_lane_coords(corrected_img)
            self.publish_lane_coords(final_img)


        else:
            final_msg = self.bridge.cv2_to_imgmsg(final_img)
            self.pub.publish(final_msg)
            self.publish_lane_coords(final_img)

        end_time = time.time()
        print(end_time-start_time)

    def middle_callback(self, msg):
        result = msg.data
        self.middle_bool = result

    def avg_lane(self, img):
        out_img = np.zeros_like(img)
        y_indices, x_indices = np.nonzero(img)[0], np.nonzero(img)[1]

        for y in np.unique(y_indices):
            x_index = x_indices[y_indices == y]
            avg_x_index = int(np.mean(x_index))
            out_img[y, avg_x_index] = 255

        return out_img

    def process_img(self, img):
        image = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
        ymax = 480
        
        x1 = int(w // 2 - 40 * 6.4)
        x2 = int(w // 2 + 20 * 6.4)

        l = 75 * 8
        tl = (215, 315)
        bl = (50, 430)
        tr = (410, 315)
        br = (565, 430)

        # tl = (215, 265)
        # bl = (50, 385)
        # tr = (410, 265)
        # br = (565, 385)

        # tl = (100, 280)
        # bl = (2, 363)
        # tr = (350, 280)
        # br = (417, 363)

        # racist
        # x1 = w // 2 - 38 * 8
        # x2 = w // 2 + 14 * 8
        # l = 60 * 8
        # tl = (100, 350)
        # bl = (2, 433)
        # tr = (350, 350)
        # br = (417, 433)

        # stopline = Int64MultiArray()
        # cv2.circle(image, tl, 5, (0, 0, 255), -1)
        # cv2.circle(image, bl, 5, (0, 0, 255), -1)
        # cv2.circle(image, tr, 5, (0, 0, 255), -1)
        # cv2.circle(image, br, 5, (0, 0, 255), -1)
        # cv2.imshow("Original", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # gamma = 0.4
        # table = [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
        # table = np.array(table, np.uint8)
        # gamma_img = cv2.LUT(img, table)

        source = np.array([bl, br, tr, tl], dtype="float32")
        destination = np.float32(
            [[x1, ymax], [x2, ymax], [x2, ymax - l], [x1, ymax - l]])
        M = cv2.getPerspectiveTransform(source, destination)
        image = cv2.warpPerspective(
            image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # return image
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary_threshold = 130
        thresh, edges = cv2.threshold(gray_img, binary_threshold, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # input_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # # binary_threshold = 185
        # binary_threshold = rospy.get_param('BINARY_THRESHOLD')
        # # print(binary_threshold)

        # ret, edges = cv2.threshold(input_image_gray, binary_threshold, 255, cv2.THRESH_BINARY)
        #edges = cv2.adaptiveThreshold(input_image_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,0)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(edges, kernel, iterations=6)
        erosion_half = erosion[erosion.shape[0] // 2:, :]
        erosion_half_resize = cv2.resize(
            erosion_half, (640, 480), interpolation=cv2.INTER_LINEAR)
        erosion_half_resize[:, 0:30] = 0
        erosion_half_resize[:, -30:] = 0
        return erosion_half_resize

    def find_lane_pixels(self, image):
        img_ratio = rospy.get_param('IMG_RATIO')
        # img_ratio = 1 / 3

        histogram = np.sum(
            image[int(image.shape[0] * (1 - img_ratio)):, :] / 255, axis=0)
        smooth_histogram = scipy.ndimage.gaussian_filter(histogram, sigma=30)
        # midpoint = np.int64(histogram.shape[0] // 2)
        maxima_indices = argrelextrema(smooth_histogram, np.greater)[0]

        if len(maxima_indices) >= 2:
            maxima_values = smooth_histogram[maxima_indices]
            maxima_values = np.sort(maxima_values)[::-1]
            maxima_indices[0] = np.where(np.isin(smooth_histogram, maxima_values[0]))[0]
            maxima_indices[1] = np.where(np.isin(smooth_histogram, maxima_values[1]))[0]

            if maxima_indices[1] > maxima_indices[0]:
                rightx_base = maxima_indices[1]
                leftx_base = maxima_indices[0]
            else:
                rightx_base = maxima_indices[0]
                leftx_base = maxima_indices[1]
            print(leftx_base, rightx_base)
            # print("max: " + str(maxima_indices))

        elif len(maxima_indices) == 1:
            # print("entered")
            # rightx_base = maxima_indices[0]
            # leftx_base = maxima_indices[0]
            if maxima_indices[0] > 320:
                rightx_base = maxima_indices[0]
                leftx_base = np.argmax(smooth_histogram[:maxima_indices[0] - 320])
            else:
                leftx_base = maxima_indices[0]
                rightx_base = np.argmax(smooth_histogram[maxima_indices[0] + 320:])

        else:
            print("NO LANES!!!!")
            return
        # numbers = np.arange(image.shape[1])  # Use np.arange to match the shape
        # plt.figure(figsize=(10, 10))
        # plt.plot(numbers, smooth_histogram)
        # Draw a vertical line at x = 320
        # plt.axvline(x=midpoint, color='r', linestyle='--')
        # plt.show()

        out_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        # cv2.imshow('Windows', image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # leftx_base = np.argmax(smooth_histogram[:midpoint])
        # rightx_base = np.argmax(smooth_histogram[midpoint:]) + midpoint

        nwindows = rospy.get_param('NWINDOWS')
        minpix = rospy.get_param('MINPIX')

        # nwindows = 10
        # margin = 50
        # minpix = 300

        window_height = np.int64(image.shape[0] // nwindows)

        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        left_lane_inds = []
        hit_left = False

        for window in range(nwindows):
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

            if len(good_left_inds) < minpix:
                if hit_left:
                    break
                else:
                    cv2.rectangle(self.img_color, (win_xleft_low, win_y_low),
                                (win_xleft_high, win_y_high), (255, 0, 0), 4)
                    continue

            else:
                hit_left = True
                leftx_current = np.int64(1 * np.mean(nonzerox[good_left_inds]))
                cv2.rectangle(self.img_color, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (255, 0, 0), 4)
                left_lane_inds.append(good_left_inds)

        rightx_current = rightx_base
        right_lane_inds = []
        hit_right = False

        for window in range(nwindows):
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            if len(good_right_inds) < minpix:
                if hit_right:
                    break
                else:
                    cv2.rectangle(self.img_color, (win_xright_low, win_y_low),
                                (win_xright_high, win_y_high), (0, 0, 255), 4)
                    continue
            cv2.rectangle(self.img_color, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 0, 255), 4)
            right_lane_inds.append(good_right_inds)

            if len(good_right_inds) > minpix:
                rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))
                hit_right = True

        left_lane_inds = np.concatenate(
            left_lane_inds) if left_lane_inds else np.array([])
        right_lane_inds = np.concatenate(
            right_lane_inds) if right_lane_inds else np.array([])

        leftx = nonzerox[left_lane_inds] if left_lane_inds.size else np.array([
        ])
        lefty = nonzeroy[left_lane_inds] if left_lane_inds.size else np.array([
        ])
        rightx = nonzerox[right_lane_inds] if right_lane_inds.size else np.array([
        ])
        righty = nonzeroy[right_lane_inds] if right_lane_inds.size else np.array([
        ])
        
        self.window_pub.publish(self.bridge.cv2_to_imgmsg(self.img_color))
        # cv2.imshow('Windows', self.img_color)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        return leftx, lefty, rightx, righty, out_img

    def search_around_poly(self, image):
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(image)

        len_left, len_right = len(leftx), len(rightx)
        print(len_left, len_right)

        if len_left:
            left_fit = np.polyfit(lefty, leftx, 2)
            left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                           left_fit[2] - self.margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                                 left_fit[1] * nonzeroy + left_fit[
                                               2] + self.margin)))
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            out_img[nonzeroy[left_lane_inds],
                    nonzerox[left_lane_inds]] = [255, 0, 0]
            # pt = (nonzerox[left_lane_inds], nonzeroy[left_lane_inds])
            # cv2.circle(out_img, pt, 1, (255, 0, 0), 1)

            

        if len_right:

            right_fit = np.polyfit(righty, rightx, 2)
            right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                            right_fit[2] - self.margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                                   right_fit[1] * nonzeroy + right_fit[
                                                2] + self.margin)))
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            out_img[nonzeroy[right_lane_inds],
                    nonzerox[right_lane_inds]] = [0, 0, 255]
            # pt = (nonzerox[right_lane_inds], nonzeroy[right_lane_inds])
            # cv2.circle(out_img, pt, 1, (0, 0, 255), 1)
        return out_img

    def get_lane_dist(self, b, r):
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

    def form_middle_lane(self, b, r):
        img = cv2.merge([b, np.zeros_like(b), r])
        for y in range(img.shape[0]):
            x_coords = np.nonzero(img[y] == 255)[0]

            if len(x_coords) >= 2:
                # Get the first two x-coordinates where img[y] == 255
                x1, x2 = x_coords[:2]
                midpoint = (x1 + x2) // 2  

                img[y, midpoint] = [0, 255, 0]
                # pt = (midpoint, y)
                # cv2.circle(img, pt, 1, (0, 255, 0), 1)

        return img

    def form_right_or_left_lane(self, b, r):
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
                        # pt = (new_x, y)
                        # cv2.circle(img, pt, 1, (255, 0, 0), 1)

        else:
            # print('Blue is longer')
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
                        # pt = (new_x, y)
                        # cv2.circle(img, pt, 1, (0, 0, 255), 1)


        return img

    def form_single_lane(self, b, r):

        if b.any():
            y_indices, x_indices = np.nonzero(b)[0], np.nonzero(b)[1]
            min_x = np.min(x_indices)
            max_x = np.max(x_indices)
            min_x_index = np.where(x_indices == min_x)[0][0]
            max_x_index = np.where(x_indices == max_x)[0][0]

            y1 = 480 - y_indices[min_x_index]
            y2 = 480 - y_indices[max_x_index]

            slope = (y2 - y1) / (max_x - min_x)

            # print(y1, y2)
            # print(min_x, max_x)

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

            print(slope)
            # print(y1, y2)
            # print(min_x, max_x)

            if slope >= 0:
                img = cv2.merge([r, np.zeros_like(r), np.zeros_like(r)])
            else:
                img = cv2.merge([np.zeros_like(r), np.zeros_like(r), r])

        return img

    def visualize_lane_detection(self, image):
        out_img = self.search_around_poly(image)

        b, g, r = cv2.split(out_img)
        b = self.avg_lane(b)
        r = self.avg_lane(r)

        if b.any() and r.any():
            dist = self.get_lane_dist(b, r)
            print("dist: " + str(dist))
            threshold_dist = rospy.get_param('THRESHOLD_DIST')
            if dist >= threshold_dist:
                final = self.form_middle_lane(b, r)
            else:
                final = self.form_right_or_left_lane(b, r)

        else:
            final = self.form_single_lane(b, r)

        return final

    def publish_lane_coords(self, image):
        b, g, r = cv2.split(image)
        b_y, b_x = np.nonzero(b)[0], np.nonzero(b)[1]

        if self.middle_bool:
            print("CALLED")
            g_y, g_x = np.array([]), np.array([])
            # Exclude green points from points
            points = np.nonzero(b | r)
        else:
            g_y, g_x = np.nonzero(g)[0], np.nonzero(g)[1]
            points = np.nonzero(image)

        r_y, r_x = np.nonzero(r)[0], np.nonzero(r)[1]
        points = np.column_stack((points[1], points[0]))

        lane_coordinates = LaneCoordinates()
        lane_coordinates.rx = r_x
        lane_coordinates.ry = r_y
        lane_coordinates.mx = g_x
        lane_coordinates.my = g_y
        lane_coordinates.lx = b_x
        lane_coordinates.ly = b_y

        points_msg = Float64MultiArray()
        points_msg.data = points.flatten()
        self.occupancy_points_pub.publish(points_msg)
        self.lane_coordinates_pub.publish(lane_coordinates)

if __name__ == '__main__':
    SlidingWindow()
    rospy.spin()
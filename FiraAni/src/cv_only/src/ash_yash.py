#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int64MultiArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import time
from cv_only.msg import LaneCoordinates
import scipy.ndimage
from threading import *
class LaneDetection(Thread):
    def __init__(self):
        self.bridge = CvBridge()
        self.j = 0
        self.left = 0

        rospy.init_node('lane_detection_node')
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        # self.stopline_pub = rospy.Publisher("/stopline", Int64MultiArray, queue_size=10)
        self.lane_coordinates_pub = rospy.Publisher("/lane_coordinates", LaneCoordinates, queue_size=10)
        self.erosion_pub_image=rospy.Publisher("/erosion_lane_image", Image, queue_size=10)
        self.lane_pub_image = rospy.Publisher("/lane_image", Image, queue_size=10)
    def image_callback(self, msg):
        start_time = time.time()

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        image = cv2.resize(cv_image, (640, 480), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
        ymax = 480
        x1 = w // 2 - 40 * 8
        x2 = w // 2 + 20 * 8
        l = 400
        tl=(190,340)
        bl=(48,420)
        tr=(391,340)
        br=(453,420)

        stopline = Int64MultiArray()
        cv2.circle(image, tl, 5, (0, 0, 255), -1)
        cv2.circle(image, bl, 5, (0, 0, 255), -1)
        cv2.circle(image, tr, 5, (0, 0, 255), -1)
        cv2.circle(image, br, 5, (0, 0, 255), -1)
        # cv2.imshow("Original", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        source = np.array([bl, br, tr, tl], dtype="float32")
        destination = np.float32([[x1, ymax], [x2, ymax], [x2, ymax - l], [x1, ymax - l]])
        M = cv2.getPerspectiveTransform(source, destination)
        image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # cv2.imshow("Original", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        input_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, edges = cv2.threshold(input_image_gray, 150, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(edges, kernel, iterations=8)
        erosion_half = erosion[erosion.shape[0] // 2:, :]
        erosion_half_resize = cv2.resize(erosion_half, (640, 480), interpolation=cv2.INTER_LINEAR)


        # erosion_half_resize = cv2.cvtColor(erosion_half_resize, cv2.COLOR_GRAY2BGR) # ADDED FOR TESTING
        # tot_img = np.concatenate([polyfitimage, erosion_half_resize, image])
        # cv2.imwrite(f'/home/umic/baggies/1/camera_image{j}.png', tot_img)

        # cv2.imshow(f'erosion{self.j}', erosion_half_resize)
        # cv2.imwrite(f'/home/umic/baggies/threshold_correct/camera_image{self.j}.png',erosion_half_resize)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        erosion_msg = self.bridge.cv2_to_imgmsg(np.array(erosion_half_resize), encoding="8UC1")
        self.erosion_pub_image.publish(erosion_msg)

        lines = cv2.HoughLinesP(erosion_half_resize, cv2.HOUGH_PROBABILISTIC, np.pi / 180, 3, minLineLength=30, maxLineGap=5)

        histogram = np.sum(erosion_half_resize / 255, axis=0)
        histogram_stop = np.sum(erosion_half_resize / 255, axis=1)

        if np.all(histogram_stop < 100):
            stopline.data = [320, 480]
        else:
            max_index = np.argmax(histogram_stop)
            stopline.data = [320, max_index]
        # self.stopline_pub.publish(stopline)

        smooth_histogram = scipy.ndimage.gaussian_filter(histogram, sigma=30)
        minima_indices = argrelextrema(smooth_histogram, np.less)[0]
        maxima_indices = argrelextrema(smooth_histogram, np.greater)[0]
        threshold = 10

        minima_to_remove = self.remove_minima(smooth_histogram, minima_indices, maxima_indices, threshold)
        mask1 = np.isin(minima_indices, minima_to_remove, invert=True)
        minima_indices = minima_indices[mask1]
        mask2 = np.isin(smooth_histogram[minima_to_remove], smooth_histogram[minima_to_remove], invert=True)
        minima_values = smooth_histogram[mask2]

        # self.plot_histogram(histogram, minima_indices, minima_values, maxima_indices, smooth_histogram)  #dont uncomment it 


        if len(minima_indices) > 1:
            self.process_three_lanes(lines, minima_indices,erosion)
        elif len(minima_indices)==1:
            self.process_two_lanes(lines, minima_indices, maxima_indices, smooth_histogram,erosion)
        else:
            self.process_single_lane(lines,erosion)

        # Publishing the lane coordinates
        lane_coordinates = LaneCoordinates()
        lane_coordinates.rx = self.y_right
        lane_coordinates.ry = self.x_right
        lane_coordinates.mx = self.y_middle
        lane_coordinates.my = self.x_middle
        lane_coordinates.lx = self.y_left
        lane_coordinates.ly = self.x_left
        self.lane_coordinates_pub.publish(lane_coordinates)
        self.rgbimage()
        end_time=time.time()
        print(end_time-start_time)
    def remove_minima(self, smooth_histogram, minima_indices, maxima_indices, threshold):
        minima_to_remove = []
        for min_index in minima_indices:
            left_indices = maxima_indices[maxima_indices < min_index]
            right_indices = maxima_indices[maxima_indices > min_index]
            if len(left_indices) == 0:
                left_indices = [0]
            if len(right_indices) == 0:
                right_indices = [630]
            if len(left_indices) > 0 and len(right_indices) > 0:
                nearest_left_max_index = left_indices[-1]
                nearest_right_max_index = right_indices[0]

                left_max_value_diff = smooth_histogram[nearest_left_max_index] - smooth_histogram[min_index]
                right_max_value_diff = smooth_histogram[nearest_right_max_index] - smooth_histogram[min_index]

                if left_max_value_diff < threshold or right_max_value_diff < threshold:
                    minima_to_remove.append(min_index)
        return minima_to_remove

    def plot_histogram(self, histogram, minima_indices, minima_values, maxima_indices, smooth_histogram):
        plt.figure(figsize=(10, 10))
        plt.plot(range(640), histogram)
        plt.scatter(minima_indices, minima_values, color='black', label='Minima Points')
        plt.scatter(maxima_indices, smooth_histogram[maxima_indices], color='blue', label='Maxima Points')
        plt.xlabel("image_xdim")
        plt.ylabel("intensity")
        plt.show()

    def rgbimage(self):
        
        polyfitimage = np.zeros((480,640,3), dtype=np.uint8) #480,640,3
        for j in range(len(self.x_left)):
            polyfitimage[int(self.x_left[j]),int(self.y_left[j])] = (0,0,255)
        for j in range(len(self.x_middle)):
            polyfitimage[int(self.x_middle[j]),int(self.y_middle[j])] = (0,255,0)
        for j in range(len(self.x_right)):
            polyfitimage[int(self.x_right[j]),int(self.y_right[j])] = (255,0,0)
        polyfitimage_msg = self.bridge.cv2_to_imgmsg(np.array(polyfitimage), encoding="bgr8")
        self.lane_pub_image.publish(polyfitimage_msg)   


    def process_three_lanes(self, lines, minima_indices,erosion):
        self.lx = []
        self.mx = []
        self.rx = []
        self.ly=[]
        self.my = []
        self.ry = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if (x1 > 0 and x2 > 0) and (x1 < minima_indices[0] and x2 < minima_indices[0]):
                    self.lx.append(x1)
                    self.lx.append(x2)
                    self.ly.append(y1)
                    self.ly.append(y2)
                elif (x1 > minima_indices[0] and x2 > minima_indices[0]) and (x1 < minima_indices[-1] and x2 < minima_indices[-1]):
                    self.mx.append(x1)
                    self.mx.append(x2)
                    self.my.append(y1)
                    self.my.append(y2)

                elif (x1 > minima_indices[-1] and x2 > minima_indices[-1]) and (x1 < 639 and x2 < 639):
                    self.rx.append(x1)
                    self.rx.append(x2)
                    self.ry.append(y1)
                    self.ry.append(y2)
        # if len(self.lx) > len(self.rx):
        #     self.y_right, self.x_right, self.y_middle, self.x_middle, self.y_left, self.x_left = self.polyfit_lane_left(self.lx, erosion)
        # else:
        self.y_right, self.x_right, self.y_middle, self.x_middle, self.y_left, self.x_left = self.polyfit_lane_right(self.ry,self.rx, erosion)

    def process_two_lanes(self, lines, minima_indices, maxima_indices, smooth_histogram,erosion):
        nearest_left_max_index = maxima_indices[maxima_indices < minima_indices[0]][-1]
        nearest_right_max_index = maxima_indices[maxima_indices > minima_indices[0]][0]
        k = smooth_histogram[nearest_left_max_index] - smooth_histogram[nearest_right_max_index]

        self.lx = []
        self.mx = []
        self.rx = []
        self.ly = []
        self.my = []
        self.ry = []
        if k > 0:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if(x1>0 and x2 > 0) and (x1<minima_indices[0] and x2 <minima_indices[0] ):
                        self.lx.append( x1)                         
                        self.lx.append( x2)
                        self.ly.append( y1)                         
                        self.ly.append( y2)
                    elif (x1>minima_indices[0] and x2 >minima_indices[0] ):
                        self.mx.append(x1)
                        self.mx.append(x2)
                        self.my.append(y1)
                        self.my.append(y2)
        else:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if(x1>0 and x2 > 0) and (x1<minima_indices[0] and x2 <minima_indices[0] ):
                        self.mx.append(x1)
                        self.mx.append(x2)
                        self.my.append(y1)
                        self.my.append(y2)
                    elif (x1>minima_indices[0] and x2 >minima_indices[0] ):
                        self.rx.append(x1)
                        self.rx.append(x2)
                        self.ry.append(y1)
                        self.ry.append(y2)
        if len(self.rx)==0:
            self.y_right, self.x_right, self.y_middle, self.x_middle, self.y_left, self.x_left=self.polyfit_lane_left(self.ly,self.lx,erosion)
            self.left=1
        else:
            self.y_right, self.x_right, self.y_middle, self.x_middle, self.y_left, self.x_left=self.polyfit_lane_right(self.ry,self.rx,erosion)
            self.left=0

    def process_single_lane(self, lines,erosion):
        self.linepoints_x = []
        self.linepoints_y= []
      
        for line in lines:
            for x1, y1, x2, y2 in line:

                self.linepoints_x.append(x1)
                self.linepoints_x.append(x2)
                self.linepoints_y.append(y1)
                self.linepoints_y.append(y2)

        if (self.left==1):
            self.y_right, self.x_right, self.y_middle, self.x_middle, self.y_left, self.x_left = self.polyfit_lane_left(self.linepoints_y,self.linepoints_x, erosion)
        else:
            self.y_right, self.x_right, self.y_middle, self.x_middle, self.y_left, self.x_left = self.polyfit_lane_right(self.linepoints_y,self.linepoints_x, erosion)

    def polyfit_lane_left(self,y_points,x_points, img):


        z = np.polynomial.polynomial.Polynomial.fit(y_points, x_points, 2)
        z_deriv = z.deriv()
        test_x_right = np.arange(-2 * img.shape[0], 2 * img.shape[0])
        test_y_right = z(test_x_right).astype("int32")
        test_z_deriv = z_deriv(test_x_right)

        test_x_left = (test_x_right - 58 * 8 * test_z_deriv / np.sqrt(1 + np.square(test_z_deriv))).astype("int32")
        test_y_left = (test_y_right + 58 * 8 / np.sqrt(1 + np.square(test_z_deriv))).astype("int32")

        test_x_middle = (test_x_right - 29 * 8 * test_z_deriv / np.sqrt(1 + np.square(test_z_deriv))).astype("int32")
        test_y_middle = (test_y_right + 29 * 8 / np.sqrt(1 + np.square(test_z_deriv))).astype("int32")

        x_right = []
        y_right = []
        x_left = []
        y_left = []
        x_middle = []
        y_middle = []

        for i in range(len(test_x_right)):
            if (test_x_right[i] >= 0 and test_x_right[i] < img.shape[0] and test_y_right[i] >= 0 and test_y_right[i] < img.shape[1]):
                x_right.append(test_x_right[i])
                y_right.append(test_y_right[i])

            if (test_x_left[i] >= 0 and test_x_left[i] < img.shape[0] and test_y_left[i] >= 0 and test_y_left[i] < img.shape[1]):
                x_left.append(test_x_left[i])
                y_left.append(test_y_left[i])

            if (test_x_middle[i] >= 0 and test_x_middle[i] < img.shape[0] and test_y_middle[i] >= 0 and test_y_middle[i] < img.shape[1]):
                x_middle.append(test_x_middle[i])
                y_middle.append(test_y_middle[i])

        return y_right, x_right, y_middle, x_middle, y_left, x_left

    def polyfit_lane_right(self, y_points,x_points, img):
       

        z = np.polynomial.polynomial.Polynomial.fit(y_points, x_points, 2)
        z_deriv = z.deriv()
        test_x_right = np.arange(-2 * img.shape[0], 2 * img.shape[0])
        test_y_right = z(test_x_right).astype("int32")
        test_z_deriv = z_deriv(test_x_right)

        test_x_left = (test_x_right + 54 * 8 * test_z_deriv / np.sqrt(1 + np.square(test_z_deriv))).astype("int32")
        test_y_left = (test_y_right - 54 * 8 / np.sqrt(1 + np.square(test_z_deriv))).astype("int32")

        test_x_middle = (test_x_right + 29 * 8 * test_z_deriv / np.sqrt(1 + np.square(test_z_deriv))).astype("int32")
        test_y_middle = (test_y_right - 29 * 8 / np.sqrt(1 + np.square(test_z_deriv))).astype("int32")

        x_right = []
        y_right = []
        x_left = []
        y_left = []
        x_middle = []
        y_middle = []

        for i in range(len(test_x_right)):
            if (test_x_right[i] >= 0 and test_x_right[i] < img.shape[0] and test_y_right[i] >= 0 and test_y_right[i] < img.shape[1]):
                x_right.append(test_x_right[i])
                y_right.append(test_y_right[i])

            if (test_x_left[i] >= 0 and test_x_left[i] < img.shape[0] and test_y_left[i] >= 0 and test_y_left[i] < img.shape[1]):
                x_left.append(test_x_left[i])
                y_left.append(test_y_left[i])

            if (test_x_middle[i] >= 0 and test_x_middle[i] < img.shape[0] and test_y_middle[i] >= 0 and test_y_middle[i] < img.shape[1]):
                x_middle.append(test_x_middle[i])
                y_middle.append(test_y_middle[i])

        return y_right, x_right, y_middle, x_middle, y_left, x_left


if __name__ == "__main__":
    lane_detection = LaneDetection()
    rospy.spin()

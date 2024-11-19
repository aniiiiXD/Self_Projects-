#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int64MultiArray
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from cv_only.msg import LaneCoordinates  
import time
j=0
left=0
def image_callback(msg):
    strt_time=time.time()

    global j,lx,rx,mx,left
    
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite(f'/home/umic/baggies/lanereal/camera_image{j}.png',cv_image)

    except CvBridgeError as e:
        print(e)
        return
    

    

# Assuming cv_image is your input image
    image = cv2.resize(cv_image, (640, 480), interpolation=cv2.INTER_AREA)

    # cv2.imshow("Original", image)

    (h, w) = (image.shape[0], image.shape[1])
    ymax = 480
    x1 = w//2 - 40*8
    x2 = w//2 + 20*8
    l = 400
    tl=(140,340)#
    bl=(3,420)
    tr=(410,340)
    br=(460,420)
    stopline = Int64MultiArray()
    cv2.circle(image,tl,5,(0,0,255),-1)
    cv2.circle(image,bl,5,(0,0,255),-1)
    cv2.circle(image,tr,5,(0,0,255),-1)
    cv2.circle(image,br,5,(0,0,255),-1)
    # cv2.circle(image,(),5,(0,0,255),-1)
    source = np.array([bl,br,tr,tl], dtype = "float32")

    # cv2.imshow("Original", image)
    # Image coordinates after undistortion
    # source = np.float32([[10., 420.], [500., 420.], [400., 320.], [180., 320.]])
    # cv2.imshow("Original", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Image coordinates without undistortion
    # source = np.float32([[19., 550.], [1004., 611.], [767., 436.], [374., 431.]])
    destination  = np.float32([[x1, ymax], [x2, ymax], [x2, ymax-l], [x1, ymax-l]])
    M = cv2.getPerspectiveTransform(source, destination)
    image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # cv2.imshow("Original", image)
    inputImage=image#remove this later
    # inputImage=cv2.resize(cv_image,(640,480))
    inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    # ret, inputimagecanny= cv2.threshold(inputImageGray, 120, 255, cv2.THRESH_BINARY)
    # edges = cv2.Canny(inputImageGray,150,230,apertureSize = 3)
    ret, edges = cv2.threshold(inputImageGray, 215, 255, cv2.THRESH_BINARY)
    # skel = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((6,6)))
    kernel = np.ones((3,3),np.uint8)
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erosion = cv2.erode(edges,kernel,iterations = 7)
    # cv2.imshow("erosion", erosion)
    erosion_half=erosion[erosion.shape[0]//2:, :]
    erosion_half_resize=cv2.resize(erosion_half,(640,480),interpolation=cv2.INTER_LINEAR)
    # erosion_half_resize=erosion_half
    erosion_half_resize[:, 0:20] = 0
    erosion_half_resize[:, 620:-1] = 0
    minLineLength = 30
    maxLineGap = 5
    line_points=[]
    mid_points=[]
    lines = cv2.HoughLinesP(erosion_half_resize,cv2.HOUGH_PROBABILISTIC, np.pi/180, 3, minLineLength,maxLineGap)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # print(len(lines))

    # cv2.imshow("erosion", erosion)
    # cv2.imshow(f'erosion{j}', erosion_half_resize)
    cv2.imwrite(f'/home/umic/baggies/threshold_correct/camera_image{j}.png',erosion_half_resize)


    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    histogram = np.sum(erosion_half_resize/255, axis=0)
    histogram_stop = np.sum(erosion_half_resize/255, axis=1)

    
    if np.all(histogram_stop < 100):
         stopline.data = [320,480]   
    else:
        max_index = np.argmax(histogram_stop)
        
        stopline.data = [320,max_index]
    stopline_pub.publish(stopline)
    

    numbers = list(range(640))
    import scipy.ndimage
    smooth_histogram = scipy.ndimage.gaussian_filter(histogram, sigma=30)
    minima_indices = argrelextrema(smooth_histogram, np.less)[0]
    minima_values = smooth_histogram[minima_indices]
    maxima_indices=argrelextrema(smooth_histogram, np.greater)[0]
    maxima_values=smooth_histogram[maxima_indices]
    threshold = 10  # Adjust this threshold according to your requirement

    minima_to_remove = []

    # for min_index in minima_indices:
    #     nearest_left_max_index = maxima_indices[maxima_indices < min_index][-1]
    #     nearest_right_max_index = maxima_indices[maxima_indices > min_index][0]

    #     left_max_value_diff = smooth_histogram[nearest_left_max_index]-smooth_histogram[min_index]  
    #     right_max_value_diff =smooth_histogram[nearest_right_max_index]- smooth_histogram[min_index]  

    #     if left_max_value_diff < threshold or right_max_value_diff < threshold:
    #         minima_to_remove.append(min_index)
    #         # print(minima_to_remove)

    for min_index in minima_indices:
        left_indices = maxima_indices[maxima_indices < min_index]
        right_indices = maxima_indices[maxima_indices > min_index]
        if len(left_indices)==0:
            left_indices=[0]
        if len(right_indices)==0:
            right_indices=[630]
        if len(left_indices) > 0 and len(right_indices) > 0:
            nearest_left_max_index = left_indices[-1]
            nearest_right_max_index = right_indices[0]

            left_max_value_diff = smooth_histogram[nearest_left_max_index] - smooth_histogram[min_index]
            right_max_value_diff = smooth_histogram[nearest_right_max_index] - smooth_histogram[min_index]

            if left_max_value_diff < threshold or right_max_value_diff < threshold:
                minima_to_remove.append(min_index)

    mask1 = np.isin(minima_indices, minima_to_remove, invert=True)
    minima_indices = minima_indices[mask1]
    mask2 = np.isin(minima_values, smooth_histogram[minima_to_remove], invert=True)
    minima_values = minima_values[mask2]


    plt.figure(figsize=(10,10))
    plt.plot(numbers,histogram)

    plt.scatter(minima_indices, minima_values, color='black', label='Minima Points')
    plt.scatter(maxima_indices, maxima_values, color='blue', label='Maxima Points')
    # plt.plot(numbers,smooth_histogram)
    plt.xlabel("image_xdim")
    plt.ylabel("intensity")
    # plt.show()
    plt.savefig(f"/home/umic/baggies/plots/fig{j}.png")
    # print(minima_indices)
    # print(maxima_indices)
    # print(maxima_values)

    # zebra_minima_diff=90

    # for i in range(len(maxima_indices)-1):
    #     maxima_diff= maxima_indices[i+1]-maxima_indices[i]
    #     if maxima_diff>zebra_minima_diff:
    #         break
    #     else:
    #         stopline.data = [320,100]

    def polyfit_lane_left(coordinates, img):
        x_points = [i[0] for i in coordinates]
        y_points = [i[1] for i in coordinates]
        
        z = np.polynomial.polynomial.Polynomial.fit(y_points, x_points, 2)
        z_deriv = z.deriv()
        test_x_right = np.arange(-2*img.shape[0], 2*img.shape[0])
        test_y_right = z(test_x_right).astype("int32")
        test_z_deriv = z_deriv(test_x_right)
        
        test_x_left = (test_x_right - 54*8*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
        
        test_y_left = (test_y_right + 54*8/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
                
        test_x_middle = (test_x_right - 27*8*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
        
        test_y_middle = (test_y_right + 27*8/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
        
        x_right = []
        y_right = []
        x_left = []
        y_left = []
        x_middle = []
        y_middle = []
        
        for i in range(len(test_x_right)):
            if(test_x_right[i] >= 0 and test_x_right[i] < (img.shape[0]) and test_y_right[i] >= 0 and test_y_right[i] < img.shape[1]):
                x_right.append(test_x_right[i] - 0)
                y_right.append(test_y_right[i])
            
            if(test_x_left[i] >= 0 and test_x_left[i] < (img.shape[0]) and test_y_left[i] >= 0 and test_y_left[i] < img.shape[1]):
                x_left.append(test_x_left[i] - 0)
                y_left.append(test_y_left[i])
            
            if(test_x_middle[i] >= 0 and test_x_middle[i] < (img.shape[0]) and test_y_middle[i] >= 0 and test_y_middle[i] < img.shape[1]):
                x_middle.append(test_x_middle[i] - 0 )
                y_middle.append(test_y_middle[i])
                
        return y_right, x_right, y_middle, x_middle, y_left, x_left
    
    def polyfit_lane_right(coordinates, img):
        x_points = [i[0] for i in coordinates]
        y_points = [i[1] for i in coordinates]
        
        z = np.polynomial.polynomial.Polynomial.fit(y_points, x_points, 2)
        z_deriv = z.deriv()
        test_x_right = np.arange(-2*img.shape[0], 2*img.shape[0])
        test_y_right = z(test_x_right).astype("int32")
        test_z_deriv = z_deriv(test_x_right)
        
        test_x_left = (test_x_right + 54*8*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
        
        test_y_left = (test_y_right - 54*8/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
                
        test_x_middle = (test_x_right + 27*8*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
        
        test_y_middle = (test_y_right - 27*8/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
                
        x_right = []
        y_right = []
        x_left = []
        y_left = []
        x_middle = []
        y_middle = []
        
        for i in range(len(test_x_right)):
            if(test_x_right[i] >= 0 and test_x_right[i] < img.shape[0] and test_y_right[i] >= 0 and test_y_right[i] < img.shape[1]):
                x_right.append(test_x_right[i] - 0)
                y_right.append(test_y_right[i])
            
            if(test_x_left[i] >= 0 and test_x_left[i] < img.shape[0] and test_y_left[i] >= 0 and test_y_left[i] < img.shape[1]):
                x_left.append(test_x_left[i] - 0)
                y_left.append(test_y_left[i])
            
            if(test_x_middle[i] >= 0 and test_x_middle[i] < img.shape[0] and test_y_middle[i] >= 0 and test_y_middle[i] < img.shape[1]):
                x_middle.append(test_x_middle[i] - 0)
                y_middle.append(test_y_middle[i])
                
        return y_left, x_left, y_middle, x_middle, y_right, x_right
    

    if len(minima_indices)>1:
        # print("3lane_case")
        lx=[]#left lane coodr
        mx=[]#middle lane coodr
        rx=[]#right lane coodr
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                if(x1>0 and x2 > 0) and (x1<minima_indices[0] and x2 <minima_indices[0] ):
                    lx.append( np.array([x1,y1], np.int32))
                    lx.append( np.array([x2,y2], np.int32))
                elif (x1>minima_indices[0] and x2 >minima_indices[0] ) and(x1<minima_indices[-1] and x2 <minima_indices[-1] ):
                    mx.append( np.array([x1,y1], np.int32))
                    mx.append( np.array([x2,y2], np.int32))
                elif  (x1>minima_indices[-1] and x2 >minima_indices[-1] )and (x1<639 and x2 <639):
                    rx.append( np.array([x1,y1], np.int32))
                    rx.append( np.array([x2,y2], np.int32))
        # if abs(len(lx)-len(rx)) <= 500:
        #     y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_left(lx,erosion)
        # # else:
        #     if(len(lx)>len(rx)):
        #         y_right, x_right, y_middle, x_middle, y_left, x_left= polyfit_lane_left(lx,erosion)

            #else:
        y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_right(rx,erosion)
        

    elif len(minima_indices)>0 and len(minima_indices)<2:
        # print("2lane_case")
        nearest_left_max_index = maxima_indices[maxima_indices < min_index][-1]
        nearest_right_max_index = maxima_indices[maxima_indices > min_index][0]
        k = smooth_histogram[nearest_left_max_index] - smooth_histogram[nearest_right_max_index]
        # k=maxima_values[0]-maxima_values[1]
        lx=[]#left lane coodr
        mx=[]# middle lane coodr
        rx=[]
        if(k>0):
            for x in range(0, len(lines)):
                for x1,y1,x2,y2 in lines[x]:
                    if(x1>0 and x2 > 0) and (x1<minima_indices[0] and x2 <minima_indices[0] ):
                        lx.append( np.array([x1,y1], np.int32))                         
                        lx.append( np.array([x2,y2], np.int32))
                    elif (x1>minima_indices[0] and x2 >minima_indices[0] ):
                        mx.append( np.array([x1,y1], np.int32))
                        mx.append( np.array([x2,y2], np.int32))
        elif(k<0):
            for x in range(0, len(lines)):
                for x1,y1,x2,y2 in lines[x]:
                    if(x1>0 and x2 > 0) and (x1<minima_indices[0] and x2 <minima_indices[0] ):
                        mx.append( np.array([x1,y1], np.int32))
                        mx.append( np.array([x2,y2], np.int32))
                    elif (x1>minima_indices[0] and x2 >minima_indices[0] ):
                        rx.append( np.array([x1,y1], np.int32))
                        rx.append( np.array([x2,y2], np.int32))

        if len(rx)==0:
            y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_left(lx,erosion)
            left=1
        else:
            y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_right(rx,erosion)
            left=0

        
            

    else:
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                # pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
                # # cv2.circle(image, [x1, y1 ], 2, (0,255,0),-1)
                # k=int((x1+x2)/2)
                # f=(y1+y2)/2
                line_points.append( np.array([x1,y1], np.int32))
                line_points.append( np.array([x2,y2], np.int32))
        #         mid_points.append([(k)])
        
        # mid=np.array(mid_points)

        # def count_white_pixels(image,x_array):
        #     height, width = image.shape[:2]
        #     counts = []
        #     for x in x_array:
        #         count = np.sum(image[:, x] == 255)  # Count white pixels in the column
        #         counts.append(count)
        #     return counts

        # # binary_image = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
        # white_pixel_counts = count_white_pixels(erosion,mid)

        # weighted_avg = sum(count * x for count, x in zip(white_pixel_counts, mid)) / sum(white_pixel_counts)

        # # print("Weighted average of white pixels:", weighted_avg)

        # height, width = erosion_half_resize.shape[:2]
        # if((weighted_avg<width/2)) :
        if (left==0):
            # print("Left")
            rx=line_points
            y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_right(rx,erosion)
        else:
            # print("right")
            lx=line_points
            y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_left(lx,erosion)


    # def polyfit_lane_left(x_points,y_points, img):
    #     # x_points = [i[0] for i in coordinates]
    #     # y_points = [i[1] for i in coordinates]
        
    #     z = np.polynomial.polynomial.Polynomial.fit(y_points, x_points, 2)
    #     z_deriv = z.deriv()
    #     test_x_right = np.arange(-2*img.shape[0], 2*img.shape[0])
    #     test_y_right = z(test_x_right).astype("int32")
    #     test_z_deriv = z_deriv(test_x_right)
        
    #     test_x_left = (test_x_right - 58*8*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
        
    #     test_y_left = (test_y_right + 58*8/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
                
    #     test_x_middle = (test_x_right - 29*8*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
        
    #     test_y_middle = (test_y_right + 29*8/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
        
    #     x_right = []
    #     y_right = []
    #     x_left = []
    #     y_left = []
    #     x_middle = []
    #     y_middle = []
        
    #     for i in range(len(test_x_right)):
    #         if(test_x_right[i] >= 0 and test_x_right[i] < (img.shape[0]) and test_y_right[i] >= 0 and test_y_right[i] < img.shape[1]):
    #             x_right.append(test_x_right[i] - 0)
    #             y_right.append(test_y_right[i])
            
    #         if(test_x_left[i] >= 0 and test_x_left[i] < (img.shape[0]) and test_y_left[i] >= 0 and test_y_left[i] < img.shape[1]):
    #             x_left.append(test_x_left[i] - 0)
    #             y_left.append(test_y_left[i])
            
    #         if(test_x_middle[i] >= 0 and test_x_middle[i] < (img.shape[0]) and test_y_middle[i] >= 0 and test_y_middle[i] < img.shape[1]):
    #             x_middle.append(test_x_middle[i] - 0 )
    #             y_middle.append(test_y_middle[i])
                
    #     return y_right, x_right, y_middle, x_middle, y_left, x_left
    
    # def polyfit_lane_right(x_points,y_points, img):
    #     # x_points = [i[0] for i in coordinates]
    #     # y_points = [i[1] for i in coordinates]
        
    #     z = np.polynomial.polynomial.Polynomial.fit(y_points, x_points, 2)
    #     z_deriv = z.deriv()
    #     test_x_right = np.arange(-2*img.shape[0], 2*img.shape[0])
    #     test_y_right = z(test_x_right).astype("int32")
    #     test_z_deriv = z_deriv(test_x_right)
        
    #     test_x_left = (test_x_right + 58*8*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
        
    #     test_y_left = (test_y_right - 58*8/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
                
    #     test_x_middle = (test_x_right + 29*8*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
        
    #     test_y_middle = (test_y_right - 29*8/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
                
    #     x_right = []
    #     y_right = []
    #     x_left = []
    #     y_left = []
    #     x_middle = []
    #     y_middle = []
        
    #     for i in range(len(test_x_right)):
    #         if(test_x_right[i] >= 0 and test_x_right[i] < img.shape[0] and test_y_right[i] >= 0 and test_y_right[i] < img.shape[1]):
    #             x_right.append(test_x_right[i] - 0)
    #             y_right.append(test_y_right[i])
            
    #         if(test_x_left[i] >= 0 and test_x_left[i] < img.shape[0] and test_y_left[i] >= 0 and test_y_left[i] < img.shape[1]):
    #             x_left.append(test_x_left[i] - 0)
    #             y_left.append(test_y_left[i])
            
    #         if(test_x_middle[i] >= 0 and test_x_middle[i] < img.shape[0] and test_y_middle[i] >= 0 and test_y_middle[i] < img.shape[1]):
    #             x_middle.append(test_x_middle[i] - 0)
    #             y_middle.append(test_y_middle[i])
                
    #     return y_left, x_left, y_middle, x_middle, y_right, x_right
    
    # def get_turn_type(x, y):
    #     pos_slope_cnt, neg_slope_cnt = 0, 0
    #     pos_slope_sum, neg_slope_sum = 0, 0
        
    #     for i in range(0, len(x) - 1, 2):
    #         x1, x2 = x[i], x[i + 1]
    #         y2, y1 = y[i], y[i + 1] # swapping as the y-axis is inverted
    #         if x1 != x2:
    #             slope = (y2 - y1) / (x2 - x1)
    #             if slope > 0:
    #                 pos_slope_cnt += 1
    #                 pos_slope_sum += slope
    #             else:
    #                 neg_slope_cnt += 1
    #                 neg_slope_sum += slope

    #     # print(pos_slope_cnt, neg_slope_cnt)

    #     if pos_slope_cnt > neg_slope_cnt:
    #         return 'right'
    #     elif pos_slope_cnt < neg_slope_cnt:
    #         return 'left'
    #     else:
    #         # higher the sum, closer is the line to the center, hence more uncertain about its direction
    #         if abs(pos_slope_sum) <= abs(neg_slope_sum): 
    #             return 'right'
    #         else:
    #             return 'left'

    # if len(minima_indices)>1:
    #     print("3lane_case")
    #     left_x, left_y, right_x, right_y,middle_x,middle_y = [], [], [], [],[],[]
    #     for x in range(0, len(lines)):
    #         for x1,y1,x2,y2 in lines[x]:
    #             if(x1>0 and x2 > 0) and (x1<minima_indices[0] and x2 <minima_indices[0] ):
    #                 left_x.append(x1)
    #                 left_x.append(x2)
    #                 left_y.append(y1) 
    #                 left_y.append(y2)
    #             elif (x1>minima_indices[0] and x2 >minima_indices[0] ) and(x1<minima_indices[-1] and x2 <minima_indices[-1] ):
    #                 middle_x.append(x1)
    #                 middle_x.append(x2)
    #                 middle_y.append(y1) 
    #                 middle_y.append(y2)
    #             elif  (x1>minima_indices[-1] and x2 >minima_indices[-1] )and (x1<639 and x2 <639):
    #                 right_x.append(x1)
    #                 right_x.append(x2)
    #                 right_y.append(y1) 
    #                 right_y.append(y2)
    #     # print(len(left_x),len(right_x))

    #     if abs(len(left_x)-len(right_x)) <= 500:
    #         y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_left(left_x,left_y,erosion)

    #     else:
    #         if(len(left_x)>len(right_x)):
    #             y_right, x_right, y_middle, x_middle, y_left, x_left= polyfit_lane_left(left_x,left_y,erosion)

    #         else:
    #             y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_right(right_x,right_y,erosion)
        

    # # hough_lines = hough_transform(processed_img, 3, 69)
    # # double lane case
    

    # elif (len(minima_indices)==1):

    #     left_x, left_y, right_x, right_y = [], [], [], []
    #     for line in lines:
    #         for x1, y1, x2, y2 in line:

    #             if (x1 < minima_indices[0] and x2 < minima_indices[0]):
    #                 left_x.append(x1)
    #                 left_x.append(x2)
    #                 left_y.append(y1) 
    #                 left_y.append(y2)

    #             elif (x1 > minima_indices[0] and x2 > minima_indices[0]):
    #                 right_x.append(x1)
    #                 right_x.append(x2)
    #                 right_y.append(y1)
    #                 right_y.append(y2)
        
    #     if len(left_x) >= len(right_x):
    #         if get_turn_type(left_x, left_y) == 'left':
    #             print('This is a LEFT turn')
    #             #use right_x,right_y as right_lane_coordinates for polyfit
    #             y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_right(right_x,right_y,erosion)
    #         else :
    #             print('This is a RIGHT turn')
    #             #use left_x,left_y as left lane_coord for ployfit
    #             y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_left(left_x,left_y,erosion)
    #     else: 
    #         if get_turn_type(right_x, right_y) == 'left':
    #             print('This is a LEFT turn')
    #             #use right_x,right_y as right_lanecoordinates for polyfit
    #             y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_right(right_x,right_y,erosion)

    #         else:
    #             print('This is a RIGHT turn')
    #             #use left_x,left_y as left lane_coord for ployfit
    #             y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_left(left_x,left_y,erosion)
            
    #     # if len(rx)==0:
    #     #     y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_left(left_x,erosion)
    #     #     left=1


    #     # else:
    #     #     y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_right(rx,erosion)
    #     #     left=0

        
            

    # else:
    #     lane_x, lane_y = [], [] # single lane case
    #     for x in range(0, len(lines)):
    #         for x1,y1,x2,y2 in lines[x]:
    #             lane_x.append(x1)
    #             lane_x.append(x2)
    #             lane_y.append(y1) 
    #             lane_y.append(y2)

    #     if get_turn_type(lane_x, lane_y) == 'left':
    #         print("This is a LEFT turn")
    #         #use right_x,right_y as right_lanecoordinates for polyfit
    #         y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_right(lane_x,lane_y,erosion)
    #     else:
    #         print("This is a RIGHT turn")  
    #         #use left_x,left_y as left lane_coord for ployfit
    #         y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_left(lane_x,lane_y,erosion)


    #     # height, width = erosion_half_resize.shape[:2]
    #     # # if((weighted_avg<width/2)) :
    #     # if (left==0):
    #     #     # print("Left")
    #     #     rx=line_points
    #     #     y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_right(rx,erosion)
    #     # else:
    #     #     # print("right")
    #     #     left_x=line_points
    #     #     y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_left(left_x,erosion)



    
    
    #have to edit polyfit functions to perform better

    polyfitimage = np.zeros((480,640,3), dtype=np.uint8) #480,640,3
    for i in range(len(x_left)):
        polyfitimage[int(x_left[i]),int(y_left[i])] = (0,0,255)
    for i in range(len(x_middle)):
        polyfitimage[int(x_middle[i]),int(y_middle[i])] = (0,255,0)
    for i in range(len(x_right)):
        polyfitimage[int(x_right[i]),int(y_right[i])] = (255,0,0)

    # erosion_half_resize = cv2.cvtCo/era_image{j}.png', tot_img)
    # cv2.imshow(f'ron{j}',polyfitimage)

    cv2.imwrite(f'/home/umic/baggies/1/camera_image{j}.png',polyfitimage)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # only for video 


    polyfitimage_msg = bridge.cv2_to_imgmsg(np.array(polyfitimage), encoding="bgr8")
    lane_pub_image.publish(polyfitimage_msg)
    erosion_msg = bridge.cv2_to_imgmsg(np.array(erosion_half_resize), encoding="8UC1")
    erosion_pub_image.publish(erosion_msg)
    # Publish lane coordinates
    lane_msg = LaneCoordinates()
    lane_msg.lx = y_right    #wrong have to change
    lane_msg.mx = y_middle
    lane_msg.rx = y_left
    lane_msg.ly = x_right
    lane_msg.my = x_middle
    lane_msg.ry = x_left
    # print(lane_msg.ly)

    lane_pub.publish(lane_msg)
    print(j)
    j=j+1
    end_time=time.time()
    print(end_time-strt_time)


def main():
    rospy.init_node('lane_detector')
    global bridge, lane_pub, left_x, mx, rx,lane_pub_image,erosion_pub_image,ly,my,ry,stopline_pub
    bridge = CvBridge()
    lane_pub = rospy.Publisher('/lane_coordinates', LaneCoordinates, queue_size=10)  # Define publisher for lane coordinates
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    
    lane_pub_image = rospy.Publisher("/lane_image", Image, queue_size=10)
    stopline_pub = rospy.Publisher("/stopline_coord", Int64MultiArray, queue_size=10)
    erosion_pub_image=rospy.Publisher("/erosion_lane_image", Image, queue_size=10)
    left_x, mx, rx,ly,my,ry = [], [], [],[],[],[]

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

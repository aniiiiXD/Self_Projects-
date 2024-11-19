#!/usr/bin/python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from beginner_tutorials.msg import LaneCoordinates  

j=0
left=0
def image_callback(msg):
    
    global lx, mx, rx,j,left
    
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        # cv2.imwrite(f'/home/umic/baggies/lanereal/camera_image{j}.png',cv_image)

    except CvBridgeError as e:
        print(e)
        return
    

    

# Assuming cv_image is your input image
    image = cv2.resize(cv_image, (640, 480), interpolation=cv2.INTER_AREA)

    # cv2.imshow("Original", image)

    (h, w) = (image.shape[0], image.shape[1])
    ymax = 480
    x1 = w//2 - 300
    x2 = w//2 + 175
    l = 400
    tl=(180,320)
    bl=(10,460)
    tr=(400,320)
    br=(500,460)
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
    # Image coordinates without undistortion
    # source = np.float32([[19., 550.], [1004., 611.], [767., 436.], [374., 431.]])
    destination  = np.float32([[x1, ymax], [x2, ymax], [x2, ymax-l], [x1, ymax-l]])
    M = cv2.getPerspectiveTransform(source, destination)
    image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    inputImage=image#remove this later
    # inputImage=cv2.resize(cv_image,(640,480))
    inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    # ret, inputimagecanny= cv2.threshold(inputImageGray, 120, 255, cv2.THRESH_BINARY)
    # edges = cv2.Canny(inputImageGray,150,230,apertureSize = 3)
    ret, edges = cv2.threshold(inputImageGray, 185, 255, cv2.THRESH_BINARY)
    # skel = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((6,6)))
    kernel = np.ones((3,3),np.uint8)
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erosion = cv2.erode(edges,kernel,iterations = 6)
    # cv2.imshow("erosion", erosion)
    erosion_half=erosion[erosion.shape[0]//2:, :]
    erosion_half_resize=cv2.resize(erosion_half,(640,480),interpolation=cv2.INTER_LINEAR)
    minLineLength = 3
    maxLineGap = 5
    line_points=[]
    mid_points=[]
    lines = cv2.HoughLinesP(erosion_half_resize,cv2.HOUGH_PROBABILISTIC, np.pi/180, 3, minLineLength,maxLineGap)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # print(len(lines))

    # cv2.imshow("erosion", erosion)
    # cv2.imshow(f'erosion{j}', erosion_half_resize)
    # cv2.imwrite(f'/home/umic/baggies/threshold_correct/camera_image{j}.png',erosion_half_resize)


    # cv2.waitKey(33)
    # cv2.destroyAllWindows()
    histogram = np.sum(erosion_half_resize/255, axis=0)
    numbers = list(range(640))
    import scipy.ndimage
    smooth_histogram = scipy.ndimage.gaussian_filter(histogram, sigma=30)
    minima_indices = argrelextrema(smooth_histogram, np.less)[0]
    minima_values = smooth_histogram[minima_indices]
    maxima_indices=argrelextrema(smooth_histogram, np.greater)[0]
    maxima_values=smooth_histogram[maxima_indices]
    threshold = 10  # Adjust this threshold according to your requirement

    minima_to_remove = []

    for min_index in minima_indices:
        nearest_left_max_index = maxima_indices[maxima_indices < min_index][-1]
        nearest_right_max_index = maxima_indices[maxima_indices > min_index][0]

        left_max_value_diff = smooth_histogram[nearest_left_max_index]-smooth_histogram[min_index]  
        right_max_value_diff =smooth_histogram[nearest_right_max_index]- smooth_histogram[min_index] 

        if left_max_value_diff < threshold or right_max_value_diff < threshold:
            minima_to_remove.append(min_index)
            # print(minima_to_remove)

    mask1 = np.isin(minima_indices, minima_to_remove, invert=True)
    minima_indices = minima_indices[mask1]
    mask2 = np.isin(minima_values, smooth_histogram[minima_to_remove], invert=True)
    minima_values = minima_values[mask2]

    plt.figure(figsize=(10,10))
    plt.plot(numbers,histogram)

    plt.scatter(minima_indices, minima_values, color='black', label='Minima Points')
    plt.scatter(maxima_indices, maxima_values, color='blue', label='Maxima Points')
    plt.plot(numbers,smooth_histogram)
    plt.xlabel("image_xdim")
    plt.ylabel("intensity")
    # plt.savefig(f"/home/umic/baggies/plots/fig{j}.png")
    # print(minima_indices)
    # print(maxima_indices)
    # print(maxima_values)

    def polyfit_lane_left(coordinates, img):
        x_points = [i[0] for i in coordinates]
        y_points = [i[1] for i in coordinates]
        
        z = np.polynomial.polynomial.Polynomial.fit(y_points, x_points, 2)
        z_deriv = z.deriv()
        test_x_right = np.arange(-2*img.shape[0], 2*img.shape[0])
        test_y_right = z(test_x_right).astype("int32")
        test_z_deriv = z_deriv(test_x_right)
        
        test_x_left = (test_x_right - 460*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
        
        test_y_left = (test_y_right + 460/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
                
        test_x_middle = (test_x_right - 230*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
        
        test_y_middle = (test_y_right + 230/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
        
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
        
        test_x_left = (test_x_right + 460*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
        
        test_y_left = (test_y_right - 460/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
                
        test_x_middle = (test_x_right + 230*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
        
        test_y_middle = (test_y_right - 230/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
                
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
        if len(lx)>len(rx):
            y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_left(lx,erosion)
        else:
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
                pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
                # cv2.circle(image, [x1, y1 ], 2, (0,255,0),-1)
                k=int((x1+x2)/2)
                # f=(y1+y2)/2
                line_points.append( np.array([x1,y1], np.int32))
                line_points.append( np.array([x2,y2], np.int32))
                mid_points.append([(k)])
        
        mid=np.array(mid_points)

        def count_white_pixels(image,x_array):
            height, width = image.shape[:2]
            counts = []
            for x in x_array:
                count = np.sum(image[:, x] == 255)  # Count white pixels in the column
                counts.append(count)
            return counts

        # binary_image = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
        white_pixel_counts = count_white_pixels(erosion,mid)

        weighted_avg = sum(count * x for count, x in zip(white_pixel_counts, mid)) / sum(white_pixel_counts)

        # print("Weighted average of white pixels:", weighted_avg)

        height, width = erosion_half_resize.shape[:2]
        # if((weighted_avg<width/2)) :
        if (left==0):
            # print("Left")
            rx=line_points
            y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_right(rx,erosion)
        else:
            # print("right")
            lx=line_points
            y_right, x_right, y_middle, x_middle, y_left, x_left=polyfit_lane_left(lx,erosion)


    
    reconstructed_image = np.zeros((480,640), dtype=np.uint8)  # Initialize an array of zeros with the same shape as the image
    
        # Set the specified points to 255
    for point in mx:
        reconstructed_image[point[1], point[0]] = 255

    for point in lx:
        reconstructed_image[point[1], point[0]] = 255

    for point in rx:
        reconstructed_image[point[1], point[0]] = 255

    # cv2.imshow('recon',reconstructed_image)
    # cv2.waitKey(0)
    
    # cv2.destroyAllWindows()


    
    
    #have to edit polyfit functions to perform better

    polyfitimage = np.zeros((480,640,3), dtype=np.uint8) #480,640,3
    for i in range(len(x_left)):
        polyfitimage[int(x_left[i]),int(y_left[i])] = (0,0,255)
    for i in range(len(x_middle)):
        polyfitimage[int(x_middle[i]),int(y_middle[i])] = (0,255,0)
    for i in range(len(x_right)):
        polyfitimage[int(x_right[i]),int(y_right[i])] = (255,0,0)


    # cv2.imshow(f'ron{j}',polyfitimage)
    # # cv2.imwrite(f'/home/umic/baggies/1/camera_image{j}.png',polyfitimage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # only for video 

    polyfitimage_msg = bridge.cv2_to_imgmsg(np.array(polyfitimage), encoding="bgr8")
    lane_pub_image.publish(polyfitimage_msg)
    erosion_msg = bridge.cv2_to_imgmsg(np.array(erosion_half_resize), encoding="8UC1")
    erosion_pub_image.publish(erosion_msg)
    # Publish lane coordinates
    lane_msg = LaneCoordinates()
    lane_msg.lx = y_right#wrong have to change
    lane_msg.mx = x_right
    lane_msg.rx = y_middle
    lane_pub.publish(lane_msg)
    print(j)
    print (left)
    j=j+1
def main():
    rospy.init_node('lane_detector', anonymous=True)
    global bridge, lane_pub, lx, mx, rx,lane_pub_image,erosion_pub_image
    bridge = CvBridge()
    lane_pub = rospy.Publisher('/lane_coordinates', LaneCoordinates, queue_size=10)  # Define publisher for lane coordinates
    rospy.Subscriber('/catvehicle/camera_front/color_image_raw', Image, image_callback)
    lane_pub_image = rospy.Publisher("/lane_image", Image, queue_size=10)
    erosion_pub_image=rospy.Publisher("/erosion_lane_image", Image, queue_size=10)
    lx, mx, rx = [], [], []

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

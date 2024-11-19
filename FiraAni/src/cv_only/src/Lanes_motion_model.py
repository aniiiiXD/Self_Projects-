#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from std_msgs.msg import Int64MultiArray
from std_msgs.msg import Float64
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from cv_only.msg import LaneCoordinates  
import message_filters
from math import cos, sin
import tf.transformations
# from occ.msg import Speed

# before running - 
# make and import a Speed custom message which contains encoder data and a timestamp

# tuning -
# fix the max time interval to be accepted between the messages: third parameter in ts
# have to convert the distance into pixels in transform image!!

j=0
left=0
img_prev=None
acc=None
vel=None
u_x=None
u_y=None
yaw=None
counter=0

prev_img = np.zeros((480, 640, 3), dtype=np.uint8)
prev_speed = 0
prev_angle = 0
prev_acc = []
img_count = 0
bridge = CvBridge()


def comparison(array1,array2):
    x_coords = array1[1]
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    c=0
    t=0
    min_x=min_x-40
    max_x=max_x+40
    for point in array2:
        t=t+1
        if(point[1]>=min_x and point[1]<=max_x):
            c=c+1
    if(t-c<50):
        return True
    return False


def show_img(img): # use for debugging
    cv2.imshow('Window', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def rotate_img(img, delta_angle):
    delta_angle *= 180 / np.pi 
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, delta_angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    return rotated_img


def shift_img(img, delta_dist):
    (h, w) = img.shape[:2]
    M = np.float32([[1, 0, 0], [0, 1, int(delta_dist*100)]])
    shifted_img = cv2.warpAffine(img, M, (w, h))
    return shifted_img


def transform_img(curr_speed, curr_angle):
    global prev_img, prev_speed, prev_angle, prev_acc, img_count

    img_count += 2

    delta_angle = curr_angle - prev_angle
    acc_along_axis = prev_acc[1] * cos(delta_angle) - prev_acc[0] * sin(delta_angle)
    delta_dist = abs((curr_speed ** 2 - prev_speed ** 2) / (2 * acc_along_axis))

    rotated_img = rotate_img(prev_img, delta_angle)
    transformed_img = shift_img(rotated_img, delta_dist * -8)

    print(delta_dist * -8)

    return transformed_img 


def process_img(cv_image):
    image = cv2.resize(cv_image, (640, 480), interpolation=cv2.INTER_AREA)
    (h, w) = (image.shape[0], image.shape[1])
    ymax = 480
    x1 = w//2 - 35*8
    x2 = w//2 + 15*8
    l = 400
    tl=(144,326)#have to set accd to reseolution rightnow it is arbitary
    bl=(7,410)
    tr=(370,333)
    br=(420,440)
    stopline = Int64MultiArray()
    cv2.circle(image,tl,5,(0,0,255),-1)
    cv2.circle(image,bl,5,(0,0,255),-1)
    cv2.circle(image,tr,5,(0,0,255),-1)
    cv2.circle(image,br,5,(0,0,255),-1)
    source = np.array([bl,br,tr,tl], dtype = "float32")

    destination  = np.float32([[x1, ymax], [x2, ymax], [x2, ymax-l], [x1, ymax-l]])
    M = cv2.getPerspectiveTransform(source, destination)
    image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    inputImageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, edges = cv2.threshold(inputImageGray, 185, 255, 
                               cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(edges,kernel,iterations = 6)
    erosion_half=erosion[erosion.shape[0]//2:, :]
    erosion_half_resize=cv2.resize(erosion_half, (640,480), 
                                   interpolation=cv2.INTER_LINEAR)
    return erosion_half, erosion_half_resize
    

def hough_transform(img, minLineLength, maxLineGap):
    return cv2.HoughLinesP(img, cv2.HOUGH_PROBABILISTIC, np.pi/180,
                          3, minLineLength, maxLineGap)


def polyfit_lane_left(coordinates, img):
    x_points = [i[0] for i in coordinates]
    y_points = [i[1] for i in coordinates]
    
    z = np.polynomial.polynomial.Polynomial.fit(y_points, x_points, 2)
    z_deriv = z.deriv()
    test_x_right = np.arange(-2*img.shape[0], 2*img.shape[0])
    test_y_right = z(test_x_right).astype("int32")
    test_z_deriv = z_deriv(test_x_right)
    
    test_x_left = (test_x_right - 400*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
    
    test_y_left = (test_y_right + 400/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
            
    test_x_middle = (test_x_right - 200*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
    
    test_y_middle = (test_y_right + 200/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
    
    x_right = []
    y_right = []
    x_left = []
    y_left = []
    x_middle = []
    y_middle = []
    r1=[]
    m1=[]
    l1=[]
    
    for i in range(len(test_x_right)):
        if(test_x_right[i] >= 0 and test_x_right[i] < img.shape[0] and test_y_right[i] >= 0 and test_y_right[i] < img.shape[1]):
            x_right.append(test_x_right[i] - 0)
            y_right.append(test_y_right[i])
            r1.append([test_x_right[i],test_y_right[i]])
        
        if(test_x_left[i] >= 0 and test_x_left[i] < img.shape[0] and test_y_left[i] >= 0 and test_y_left[i] < img.shape[1]):
            x_left.append(test_x_left[i] - 0)
            y_left.append(test_y_left[i])
            l1.append([test_x_left[i],test_y_left[i]])
        
        if(test_x_middle[i] >= 0 and test_x_middle[i] < img.shape[0] and test_y_middle[i] >= 0 and test_y_middle[i] < img.shape[1]):
            x_middle.append(test_x_middle[i] - 0)
            y_middle.append(test_y_middle[i])
            m1.append([test_x_middle[i],test_y_middle[i]])
            
    return y_right, x_right, y_middle, x_middle, y_left, x_left,r1,m1,l1


def polyfit_lane_right(coordinates, img):
    x_points = [i[0] for i in coordinates]
    y_points = [i[1] for i in coordinates]
    
    z = np.polynomial.polynomial.Polynomial.fit(y_points, x_points, 2)
    z_deriv = z.deriv()
    test_x_right = np.arange(-2*img.shape[0], 2*img.shape[0])
    test_y_right = z(test_x_right).astype("int32")
    test_z_deriv = z_deriv(test_x_right)
    
    test_x_left = (test_x_right + 400*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
    
    test_y_left = (test_y_right - 400/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
            
    test_x_middle = (test_x_right + 200*test_z_deriv / np.sqrt(1+np.square(test_z_deriv))).astype("int32")
    
    test_y_middle = (test_y_right - 200/np.sqrt(1+np.square(test_z_deriv))).astype("int32")
            
    x_right = []
    y_right = []
    x_left = []
    y_left = []
    x_middle = []
    y_middle = []
    r1=[]
    m1=[]
    l1=[]
    
    for i in range(len(test_x_right)):
        if(test_x_right[i] >= 0 and test_x_right[i] < img.shape[0] and test_y_right[i] >= 0 and test_y_right[i] < img.shape[1]):
            x_right.append(test_x_right[i] - 0)
            y_right.append(test_y_right[i])
            r1.append([test_x_right[i],test_y_right[i]])
        
        if(test_x_left[i] >= 0 and test_x_left[i] < img.shape[0] and test_y_left[i] >= 0 and test_y_left[i] < img.shape[1]):
            x_left.append(test_x_left[i] - 0)
            y_left.append(test_y_left[i])
            l1.append([test_x_left[i],test_y_left[i]])
        
        if(test_x_middle[i] >= 0 and test_x_middle[i] < img.shape[0] and test_y_middle[i] >= 0 and test_y_middle[i] < img.shape[1]):
            x_middle.append(test_x_middle[i] - 0)
            y_middle.append(test_y_middle[i])
            m1.append([test_x_middle[i],test_y_middle[i]])
            
    return y_left, x_left, y_middle, x_middle, y_right, x_right,r1,m1,l1


def count_white_pixels(image,x_array):
    height, width = image.shape[:2]
    counts = []
    for x in x_array:
        count = np.sum(image[:, x] == 255)  # Count white pixels in the column
        counts.append(count)
    return counts


def encoder_callback(msg1):
    global encoder_data
    encoder_data = msg1

def call(msg):
    img=bridge.imgmsg_to_cv2(msg,"bgr8")
    cv2.imshow("im",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def callback(imu_msg, img_msg):
    global encoder_data
    global lx, mx, rx,j,left,yaw,vel,acc,u_x,u_y
    global prev_img, prev_speed, prev_angle, prev_acc, img_count, counter
    folder_path="/home/yajan/cv_ws/src/occ/src/Images"

    # encoder_msg = encoder_data
    encoder_msg = 0.04

    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except CvBridgeError as e:
        print(e)
        return

    # acc = [
    #         imu_msg.linear_acceleration.x,
    #         imu_msg.linear_acceleration.y,
    #         imu_msg.linear_acceleration.z
    # ]    
    acc = [
            imu_msg.linear_acceleration.y,
            -imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.z
    ]   
    quaternion = [
        imu_msg.orientation.x,
        imu_msg.orientation.y,
        imu_msg.orientation.z,
        imu_msg.orientation.w
    ]
    angles = tf.transformations.euler_from_quaternion(quaternion)
    angle = angles[2]
    speed = encoder_msg

    erosion_half, erosion_half_resize = process_img(cv_image)
    # cv2.imshow("ipmed",erosion_half_resize)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # b,g,r=cv2.split(erosion_half_resize)
    # b=np.nonzero(b)
    # g=np.nonzero(g)
    # r=np.nonzero(r)
    transformed_img = np.zeros((480, 640, 3), dtype=np.uint8)
    if prev_acc and counter==1:
        transformed_img = transform_img(speed, angle)
        b,g,r=cv2.split(transformed_img)
    image_name = 'image' + str(img_count) + '.png'
    transformed_name = 'image ' + str(img_count + 1) + '.png'
    # cv2.imwrite(image_name, prev_img)
    # cv2.imwrite(transformed_name, transformed_img)
    # prev_img = erosion_half_resize
    prev_speed = speed
    prev_angle = angle
    prev_acc = acc

    print(prev_acc, prev_angle, prev_speed)

    line_points=[]
    lines = hough_transform(erosion_half_resize, 3, 5)
    histogram = np.sum(erosion_half_resize/255, axis=0)

    # histogram_stop = np.sum(erosion_half_resize/255, axis=1)
    # if np.all(histogram_stop < 100):
    #     stopline.data = [320,480]
    # else:
    #     max_index = np.argmax(histogram_stop)
    #     stopline.data = [320,max_index]
    # stopline_pub.publish(stopline)

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
        left_indices = maxima_indices[maxima_indices < min_index]
        right_indices = maxima_indices[maxima_indices > min_index]
        if left_indices.size > 0 and right_indices.size > 0:
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
    plt.plot(numbers,smooth_histogram)
    plt.xlabel("image_xdim")
    plt.ylabel("intensity")
    # plt.savefig('/home/yajan/cv_ws/src/occ/src/Images/plot.png')
    
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
            y_right, x_right, y_middle, x_middle, y_left, x_left,r1,m1,l1=polyfit_lane_left(lx,erosion_half_resize)
        else:
            y_right, x_right, y_middle, x_middle, y_left, x_left,r1,m1,l1=polyfit_lane_right(rx,erosion_half_resize)
        if(counter==1):        
            if b.size == 0:
                if comparison(g,lx) and comparison(r,mx):
                    rx=m1
                    mx=l1
                    lx=None
                    # transformed_name = 'left turn 2 lanes' + str(img_count + 1) + '.png'
                    # image_name='left turn 2 lanes and one garbage on right' + str(img_count ) + '.png'
            elif b.size !=0:
                if(comparison(b,lx)):
                    # transformed_name = '3 lanes' + str(img_count + 1) + '.png'
                    # image_name='3 lanes' + str(img_count ) + '.png'
                    return
                elif comparison(b,mx) and comparison(g,rx):
                    lx=m1
                    mx=r1
                    rx=None
                    # transformed_name = 'right turn 2 lanes' + str(img_count + 1) + '.png'
                    # image_name='right turn 2 lanes and one garbage on left' + str(img_count ) + '.png'
        if len(lx)>len(rx):
            y_right, x_right, y_middle, x_middle, y_left, x_left,r1,m1,l1=polyfit_lane_left(lx,erosion_half_resize)
        else:
            y_right, x_right, y_middle, x_middle, y_left, x_left,r1,m1,l1=polyfit_lane_right(rx,erosion_half_resize)
        


    elif len(minima_indices)==1:
        # print("2lane_case")
        nearest_left_max_index = maxima_indices[maxima_indices < min_index][-1]
        nearest_right_max_index = maxima_indices[maxima_indices > min_index][0]
        k = smooth_histogram[nearest_left_max_index] - smooth_histogram[nearest_right_max_index]
        # k=maxima_values[0]-maxima_values[1]
        lx=[]#left lane coodr
        mx=[]# middle lane coodr
        rx=[]
        if(nearest_right_max_index-nearest_left_max_index>450):  # basically 60 *8=480
            if(x1>0 and x2 > 0) and (x1<minima_indices[0] and x2 <minima_indices[0] ):
                lx.append( np.array([x1,y1], np.int32))                         
                lx.append( np.array([x2,y2], np.int32))
            elif (x1>minima_indices[0] and x2 >minima_indices[0] ):
                rx.append( np.array([x1,y1], np.int32))
                rx.append( np.array([x2,y2], np.int32))
        else:
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
                        if(x1>0 and x2 > 0) and (x1<minima_indices[0] and x2 <minima_indices[0]):
                            mx.append( np.array([x1,y1], np.int32))
                            mx.append( np.array([x2,y2], np.int32))
                        elif (x1>minima_indices[0] and x2 >minima_indices[0] ):
                            rx.append( np.array([x1,y1], np.int32))
                            rx.append( np.array([x2,y2], np.int32))
        # if len(rx)==0:
        #     y_right, x_right, y_middle, x_middle, y_left, x_left,r1,m1,l1=polyfit_lane_left(lx,erosion_half_resize)
        #     left=1
        # else:
        #     y_right, x_right, y_middle, x_middle, y_left, x_left,r1,m1,l1=polyfit_lane_right(rx,erosion_half_resize)
        #     left=0
            if counter==1:
                if b.size ==0:
                    if comparison(g,mx) and comparison(r,rx):
                        # transformed_name = 'left turn 2 lanes' + str(img_count + 1) + '.png'
                        # image_name='left turn 2 lanes ' + str(img_count ) + '.png'
                        return
                    else:
                        if comparison(r,mx):
                            rx=mx

                            mx=None
                            lx=None
                            # transformed_name = 'left turn 1 lanes' + str(img_count + 1) + '.png'
                            # image_name='left turn 1 lanes and one garbage on left' + str(img_count ) + '.png'
                        elif comparison(r,lx):
                            rx=lx
                            mx=None
                            lx=None
                            # transformed_name = 'right turn 2 lanes' + str(img_count + 1) + '.png'
                            # image_name='right turn 2 lanes and one garbage on left' + str(img_count ) + '.png'
                else:
                    if comparison(g,mx) and comparison(b,lx):
                        # transformed_name = 'right turn 2 lanes' + str(img_count + 1) + '.png'
                        # image_name='right turn 2 lanes ' + str(img_count ) + '.png'
                        return
                    else:
                        if comparison(b,mx):
                            lx=mx
                            mx=None
                            rx=None
                            # transformed_name = 'right turn 1 lanes' + str(img_count + 1) + '.png'
                            # image_name='right turn 1 lanes and one garbage on right' + str(img_count ) + '.png'
                        elif comparison(b,rx):
                            lx=rx
                            mx=None
                            rx=None
                            # transformed_name = 'left turn 2 lanes' + str(img_count + 1) + '.png'
                            # image_name='left turn 2 lanes and one garbage on right' + str(img_count ) + '.png'

        if len(rx)==0:
            y_right, x_right, y_middle, x_middle, y_left, x_left,r1,m1,l1=polyfit_lane_left(lx,erosion_half_resize)
            left=1
        else:
            y_right, x_right, y_middle, x_middle, y_left, x_left,r1,m1,l1=polyfit_lane_right(rx,erosion_half_resize)
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

        # binary_image = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
        # white_pixel_counts = count_white_pixels(erosion,mid)

        # weighted_avg = sum(count * x for count, x in zip(white_pixel_counts, mid)) / sum(white_pixel_counts)

        # print("Weighted average of white pixels:", weighted_avg)

        # height, width = erosion_half_resize.shape[:2]
        if counter==1 :
            if  b.size ==0:
                rx=line_points
                y_right, x_right, y_middle, x_middle, y_left, x_left,r1,m1,l1=polyfit_lane_right(rx,erosion_half_resize)
                # transformed_name = 'left turn' + str(img_count + 1) + '.png'
                # image_name='left turn ' + str(img_count ) + '.png'
            else:
                lx=line_points
                y_right, x_right, y_middle, x_middle, y_left, x_left,r1,m1,l1=polyfit_lane_left(lx,erosion_half_resize)
                # transformed_name = 'right turn' + str(img_count + 1) + '.png'
                # image_name='right turn ' + str(img_count ) + '.png'
        else:
            rx=line_points
            y_right, x_right, y_middle, x_middle, y_left, x_left,r1,m1,l1=polyfit_lane_right(rx,erosion_half_resize)

    reconstructed_image = np.zeros((480,640), dtype=np.uint8)  # Initialize an array of zeros with the same shape as the image
    
    # Set the specified points to 255
    for point in mx:
        reconstructed_image[point[1], point[0]] = 255

    for point in lx:
        reconstructed_image[point[1], point[0]] = 255

    for point in rx:
        reconstructed_image[point[1], point[0]] = 255

    polyfitimage = np.zeros((480,640,3), dtype=np.uint8) #480,640,3
    for i in range(len(x_left)):
        polyfitimage[int(x_left[i]),int(y_left[i])] = (0,0,255)
    for i in range(len(x_middle)):
        polyfitimage[int(x_middle[i]),int(y_middle[i])] = (0,255,0)
    for i in range(len(x_right)):
        polyfitimage[int(x_right[i]),int(y_right[i])] = (255,0,0)
    
    if counter==1:
        image_path=folder_path+image_name
        transformed_path=folder_path+transformed_name
        cv2.imwrite(image_path, erosion_half_resize)
        cv2.imwrite(transformed_path, transformed_img)
    prev_img=polyfitimage
    counter=1
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

    lane_pub.publish(lane_msg)
    print(j)
    print (left)
    j += 1


def main():
    rospy.init_node('lane_detector')    
    global bridge, lane_pub, lx, mx, rx, lane_pub_image, erosion_pub_image, ly, my, ry, acc, yaw, vel, u_x, u_y, img_prev
    # global stopline_pub
    bridge = CvBridge()
    # image_s=rospy.Subscriber('/camera/color/image_raw', Image,call)
    lane_pub = rospy.Publisher('/lane_coordinates', LaneCoordinates, queue_size=10)  # Define publisher for lane coordinates
    imu_sub = message_filters.Subscriber('/imu/data', Imu)
    encoder_sub = rospy.Subscriber('/odo', Float64, encoder_callback)

    img_sub = message_filters.Subscriber('/camera/color/image_raw', Image)

    time_interval = 0.5

    ts = message_filters.ApproximateTimeSynchronizer([imu_sub, img_sub], 20, time_interval,
                                                     allow_headerless=False)
    ts.registerCallback(callback)
   
    lane_pub_image = rospy.Publisher("/lane_image", Image, queue_size=10)
    # stopline_pub = rospy.Publisher("/stopline_coord", Int64MultiArray, queue_size=10)
    erosion_pub_image=rospy.Publisher("/erosion_lane_image", Image, queue_size=10)
    lx, mx, rx,ly,my,ry = [], [], [], [], [], []

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


# import rospy
# from sensor_msgs.msg import Image
# from sensor_msgs.msg import Imu
# from std_msgs.msg import Int64MultiArray
# from std_msgs.msg import Float64
# from cv_bridge import CvBridge, CvBridgeError
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import argrelextrema
# from occ.msg import LaneCoordinates
# import message_filters
# from math import cos, sin
# import tf.transformations
# import scipy.ndimage

# # tuning -
# # fix the max time interval to be accepted between the messages: third parameter in ts

# j = 0
# left = 0
# acc = None
# vel = None
# u_x = None
# u_y = None
# yaw = None
# flag = 0

# prev_img = np.zeros((480, 640, 3), dtype=np.uint8)
# prev_speed = 0
# prev_angle = 0
# prev_acc = []
# img_count = 0
# bridge = CvBridge()


# def comparison(array1, array2):
#     x_coords = array1[1]
#     min_x = np.min(x_coords)
#     max_x = np.max(x_coords)
#     c = 0
#     t = 0
#     min_x = min_x-40
#     max_x = max_x+40
#     for point in array2:
#         t = t+1
#         if (point[1] >= min_x and point[1] <= max_x and min_x >= 40 and max_x < 600):
#             c = c+1
#     if (t-c < 50):
#         return True
#     return False


# def show_img(img):  # use for debugging
#     cv2.imshow('Window', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def transform_img(curr_speed, curr_angle):
#     global prev_img, prev_speed, prev_angle, prev_acc, img_count

#     img_count += 2

#     delta_angle = curr_angle - prev_angle
#     acc_along_axis = prev_acc[1] * cos(delta_angle) - prev_acc[0] * sin(delta_angle)
#     delta_dist = abs(curr_speed ** 2 - prev_speed ** 2) / (2 * acc_along_axis)

#     delta_angle *= 180 / np.pi
#     (h, w) = prev_img.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, delta_angle, 1.0)
#     rotated_img = cv2.warpAffine(prev_img, M, (w, h))


# def process_img(cv_image):
#     image = cv2.resize(cv_image, (640, 480), interpolation=cv2.INTER_AREA)
#     (h, w) = (image.shape[0], image.shape[1])
#     ymax = 480
#     x1 = w//2 - 35*8
#     x2 = w//2 + 15*8
#     l = 400
#     tl = (144, 326)  # have to set accd to reseolution rightnow it is arbitary
#     bl = (7, 410)
#     tr = (370, 333)
#     br = (420, 440)
#     stopline = Int64MultiArray()
#     cv2.circle(image, tl, 5, (0, 0, 255), -1)
#     cv2.circle(image, bl, 5, (0, 0, 255), -1)
#     cv2.circle(image, tr, 5, (0, 0, 255), -1)
#     cv2.circle(image, br, 5, (0, 0, 255), -1)
#     source = np.array([bl, br, tr, tl], dtype="float32")

#     destination = np.float32(
#         [[x1, ymax], [x2, ymax], [x2, ymax-l], [x1, ymax-l]])
#     M = cv2.getPerspectiveTransform(source, destination)
#     image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR,
#                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

#     inputImageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ret, edges = cv2.threshold(inputImageGray, 185, 255,
#                                cv2.THRESH_BINARY)
#     kernel = np.ones((3, 3), np.uint8)
#     erosion = cv2.erode(edges, kernel, iterations=6)
#     erosion_half = erosion[erosion.shape[0]//2:, :]
#     erosion_half_resize = cv2.resize(erosion_half, (640, 480),
#                                      interpolation=cv2.INTER_LINEAR)
#     return erosion_half_resize


# def hough_transform(img, minLineLength, maxLineGap):
#     return cv2.HoughLinesP(img, cv2.HOUGH_PROBABILISTIC, np.pi/180,
#                            3, minLineLength, maxLineGap)


# def polyfit_lane_left(coordinates, img):
#     x_points = [i[0] for i in coordinates]
#     y_points = [i[1] for i in coordinates]

#     z = np.polynomial.polynomial.Polynomial.fit(y_points, x_points, 2)
#     z_deriv = z.deriv()
#     test_x_right = np.arange(-2*img.shape[0], 2*img.shape[0])
#     test_y_right = z(test_x_right).astype("int32")
#     test_z_deriv = z_deriv(test_x_right)

#     test_x_left = (test_x_right - 400*test_z_deriv /
#                    np.sqrt(1+np.square(test_z_deriv))).astype("int32")

#     test_y_left = (test_y_right + 400 /
#                    np.sqrt(1+np.square(test_z_deriv))).astype("int32")

#     test_x_middle = (test_x_right - 200*test_z_deriv /
#                      np.sqrt(1+np.square(test_z_deriv))).astype("int32")

#     test_y_middle = (test_y_right + 200 /
#                      np.sqrt(1+np.square(test_z_deriv))).astype("int32")

#     x_right = []
#     y_right = []
#     x_left = []
#     y_left = []
#     x_middle = []
#     y_middle = []
#     r1 = []
#     m1 = []
#     l1 = []

#     for i in range(len(test_x_right)):
#         if (test_x_right[i] >= 0 and test_x_right[i] < img.shape[0] and test_y_right[i] >= 0 and test_y_right[i] < img.shape[1]):
#             x_right.append(test_x_right[i] - 0)
#             y_right.append(test_y_right[i])
#             r1.append([test_x_right[i], test_y_right[i]])

#         if (test_x_left[i] >= 0 and test_x_left[i] < img.shape[0] and test_y_left[i] >= 0 and test_y_left[i] < img.shape[1]):
#             x_left.append(test_x_left[i] - 0)
#             y_left.append(test_y_left[i])
#             l1.append([test_x_left[i], test_y_left[i]])

#         if (test_x_middle[i] >= 0 and test_x_middle[i] < img.shape[0] and test_y_middle[i] >= 0 and test_y_middle[i] < img.shape[1]):
#             x_middle.append(test_x_middle[i] - 0)
#             y_middle.append(test_y_middle[i])
#             m1.append([test_x_middle[i], test_y_middle[i]])

#     return y_right, x_right, y_middle, x_middle, y_left, x_left, r1, m1, l1


# def polyfit_lane_right(coordinates, img):
#     x_points = [i[0] for i in coordinates]
#     y_points = [i[1] for i in coordinates]

#     z = np.polynomial.polynomial.Polynomial.fit(y_points, x_points, 2)
#     z_deriv = z.deriv()
#     test_x_right = np.arange(-2*img.shape[0], 2*img.shape[0])
#     test_y_right = z(test_x_right).astype("int32")
#     test_z_deriv = z_deriv(test_x_right)

#     test_x_left = (test_x_right + 400*test_z_deriv /
#                    np.sqrt(1+np.square(test_z_deriv))).astype("int32")

#     test_y_left = (test_y_right - 400 /
#                    np.sqrt(1+np.square(test_z_deriv))).astype("int32")

#     test_x_middle = (test_x_right + 200*test_z_deriv /
#                      np.sqrt(1+np.square(test_z_deriv))).astype("int32")

#     test_y_middle = (test_y_right - 200 /
#                      np.sqrt(1+np.square(test_z_deriv))).astype("int32")

#     x_right = []
#     y_right = []
#     x_left = []
#     y_left = []
#     x_middle = []
#     y_middle = []
#     r1 = []
#     m1 = []
#     l1 = []

#     for i in range(len(test_x_right)):
#         if (test_x_right[i] >= 0 and test_x_right[i] < img.shape[0] and test_y_right[i] >= 0 and test_y_right[i] < img.shape[1]):
#             x_right.append(test_x_right[i] - 0)
#             y_right.append(test_y_right[i])
#             r1.append([test_x_right[i], test_y_right[i]])

#         if (test_x_left[i] >= 0 and test_x_left[i] < img.shape[0] and test_y_left[i] >= 0 and test_y_left[i] < img.shape[1]):
#             x_left.append(test_x_left[i] - 0)
#             y_left.append(test_y_left[i])
#             l1.append([test_x_left[i], test_y_left[i]])

#         if (test_x_middle[i] >= 0 and test_x_middle[i] < img.shape[0] and test_y_middle[i] >= 0 and test_y_middle[i] < img.shape[1]):
#             x_middle.append(test_x_middle[i] - 0)
#             y_middle.append(test_y_middle[i])
#             m1.append([test_x_middle[i], test_y_middle[i]])

#     return y_left, x_left, y_middle, x_middle, y_right, x_right, r1, m1, l1


# def count_white_pixels(image, x_array):
#     height, width = image.shape[:2]
#     counts = []
#     for x in x_array:
#         count = np.sum(image[:, x] == 255)  # Count white pixels in the column
#         counts.append(count)
#     return counts


# def encoder_callback(msg1):
#     global encoder_data
#     encoder_data = msg1


# def call(msg):
#     img = bridge.imgmsg_to_cv2(msg, "bgr8")
#     cv2.imshow("im", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def callback(imu_msg, img_msg):
#     global encoder_data
#     global lx, mx, rx, j, left
#     global prev_img, prev_speed, prev_angle, prev_acc, img_count, flag
#     folder_path = "/home/yuvi/Documents/cv_ws/images"

#     encoder_msg = encoder_data

#     try:
#         cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
#     except CvBridgeError as e:
#         print(e)
#         return

#     acc = [
#         imu_msg.linear_acceleration.x,
#         imu_msg.linear_acceleration.y,
#         imu_msg.linear_acceleration.z
#     ]
#     quaternion = [
#         imu_msg.orientation.x,
#         imu_msg.orientation.y,
#         imu_msg.orientation.z,
#         imu_msg.orientation.w
#     ]
#     angles = tf.transformations.euler_from_quaternion(quaternion)
#     angle = angles[2]
#     speed = encoder_msg.data

#     erosion_half_resize = process_img(cv_image)

#     transformed_img = np.zeros((480, 640, 3), dtype=np.uint8)

#     if prev_acc and flag == 1:
#         transformed_img = transform_img(speed, angle)
#         b, g, r = cv2.split(transformed_img)
#     # image_name = 'image' + str(img_count) + '.png'
#     # transformed_name = 'image' + str(img_count + 1) + '.png'
#     # cv2.imwrite(image_name, prev_img)
#     # cv2.imwrite(transformed_name, transformed_img)
#     # prev_img = erosion_half_resize
#     prev_speed = speed
#     prev_angle = angle
#     prev_acc = acc

#     line_points = []
#     lines = hough_transform(erosion_half_resize, 3, 5)
#     histogram = np.sum(erosion_half_resize/255, axis=0)

#     # histogram_stop = np.sum(erosion_half_resize/255, axis=1)
#     # if np.all(histogram_stop < 100):
#     #     stopline.data = [320,480]
#     # else:
#     #     max_index = np.argmax(histogram_stop)
#     #     stopline.data = [320,max_index]
#     # stopline_pub.publish(stopline)

#     numbers = list(range(640))
#     smooth_histogram = scipy.ndimage.gaussian_filter(histogram, sigma=30)
#     minima_indices = argrelextrema(smooth_histogram, np.less)[0]
#     minima_values = smooth_histogram[minima_indices]
#     maxima_indices = argrelextrema(smooth_histogram, np.greater)[0]
#     maxima_values = smooth_histogram[maxima_indices]
#     threshold = 10  # Adjust this threshold according to your requirement

#     minima_to_remove = []

#     for min_index in minima_indices:
#         left_indices = maxima_indices[maxima_indices < min_index]
#         right_indices = maxima_indices[maxima_indices > min_index]
#         if left_indices.size > 0 and right_indices.size > 0:
#             nearest_left_max_index = left_indices[-1]
#             nearest_right_max_index = right_indices[0]

#             left_max_value_diff = smooth_histogram[nearest_left_max_index] - smooth_histogram[min_index]
#             right_max_value_diff = smooth_histogram[nearest_right_max_index] - smooth_histogram[min_index]

#             if left_max_value_diff < threshold or right_max_value_diff < threshold:
#                 minima_to_remove.append(min_index)

#     mask1 = np.isin(minima_indices, minima_to_remove, invert=True)
#     minima_indices = minima_indices[mask1]
#     mask2 = np.isin(
#         minima_values, smooth_histogram[minima_to_remove], invert=True)
#     minima_values = minima_values[mask2]

#     plt.figure(figsize=(10, 10))
#     plt.plot(numbers, histogram)

#     plt.scatter(minima_indices, minima_values,
#                 color='black', label='Minima Points')
#     plt.scatter(maxima_indices, maxima_values,
#                 color='blue', label='Maxima Points')
#     plt.plot(numbers, smooth_histogram)
#     plt.xlabel("image_xdim")
#     plt.ylabel("intensity")
#     # plt.savefig('/home/yajan/cv_ws/src/occ/src/Images/plot.png')
#     # splittable_image = split_image()
    
#     # cv2.imwrite('b.png', b)
#     # cv2.imwrite('g.png', g)
#     # cv2.imwrite('r.png', r)

#     if len(minima_indices) > 1:
#         # print("3lane_case")
#         lx = []  # left lane coodr
#         mx = []  # middle lane coodr
#         rx = []  # right lane coodr
#         for x in range(0, len(lines)):
#             for x1, y1, x2, y2 in lines[x]:
#                 if (x1 > 0 and x2 > 0) and (x1 < minima_indices[0] and x2 < minima_indices[0]):
#                     lx.append(np.array([x1, y1], np.int32))
#                     lx.append(np.array([x2, y2], np.int32))
#                 elif (x1 > minima_indices[0] and x2 > minima_indices[0]) and (x1 < minima_indices[-1] and x2 < minima_indices[-1]):
#                     mx.append(np.array([x1, y1], np.int32))
#                     mx.append(np.array([x2, y2], np.int32))
#                 elif (x1 > minima_indices[-1] and x2 > minima_indices[-1]) and (x1 < 639 and x2 < 639):
#                     rx.append(np.array([x1, y1], np.int32))
#                     rx.append(np.array([x2, y2], np.int32))
#         if len(lx) > len(rx):
#             y_right, x_right, y_middle, x_middle, y_left, x_left, r1, m1, l1 = polyfit_lane_left(
#                 lx, erosion_half_resize)
#         else:
#             y_right, x_right, y_middle, x_middle, y_left, x_left, r1, m1, l1 = polyfit_lane_right(
#                 rx, erosion_half_resize)
#         if (flag == 1):
#             if isinstance(b, np.ndarray) and b.size == 0:
#                 if comparison(g, l1) and comparison(r, m1):
#                     rx = m1
#                     mx = l1
#                     lx = None
#                     transformed_name = 'left turn 2 lanes' + \
#                         str(img_count + 1) + '.png'
#                     image_name = 'left turn 2 lanes and one garbage on right' + \
#                         str(img_count) + '.png'
#             elif isinstance(b, np.ndarray) and b.size != 0:
#                 if (comparison(b, l1)):
#                     transformed_name = '3 lanes' + str(img_count + 1) + '.png'
#                     image_name = '3 lanes' + str(img_count) + '.png'
#                     return
#                 elif comparison(b, m1) and comparison(g, r1):
#                     lx = m1
#                     mx = r1
#                     rx = None
#                     transformed_name = 'right turn 2 lanes' + \
#                         str(img_count + 1) + '.png'
#                     image_name = 'right turn 2 lanes and one garbage on left' + \
#                         str(img_count) + '.png'
#         if len(lx) > len(rx):
#             y_right, x_right, y_middle, x_middle, y_left, x_left, r1, m1, l1 = polyfit_lane_left(
#                 lx, erosion_half_resize)
#         else:
#             y_right, x_right, y_middle, x_middle, y_left, x_left, r1, m1, l1 = polyfit_lane_right(
#                 rx, erosion_half_resize)

#     elif len(minima_indices) == 1:
#         # print("2lane_case")
#         nearest_left_max_index = maxima_indices[maxima_indices < min_index][-1]
#         nearest_right_max_index = maxima_indices[maxima_indices > min_index][0]
#         k = smooth_histogram[nearest_left_max_index] - smooth_histogram[nearest_right_max_index]
#         # k=maxima_values[0]-maxima_values[1]
#         lx = []  # left lane coodr
#         mx = []  # middle lane coodr
#         rx = []
#         if (k > 0):
#             for x in range(0, len(lines)):
#                 for x1, y1, x2, y2 in lines[x]:
#                     if (x1 > 0 and x2 > 0) and (x1 < minima_indices[0] and x2 < minima_indices[0]):
#                         lx.append(np.array([x1, y1], np.int32))
#                         lx.append(np.array([x2, y2], np.int32))
#                     elif (x1 > minima_indices[0] and x2 > minima_indices[0]):
#                         mx.append(np.array([x1, y1], np.int32))
#                         mx.append(np.array([x2, y2], np.int32))
#         elif (k < 0):
#             for x in range(0, len(lines)):
#                 for x1, y1, x2, y2 in lines[x]:
#                     if (x1 > 0 and x2 > 0) and (x1 < minima_indices[0] and x2 < minima_indices[0]):
#                         mx.append(np.array([x1, y1], np.int32))
#                         mx.append(np.array([x2, y2], np.int32))
#                     elif (x1 > minima_indices[0] and x2 > minima_indices[0]):
#                         rx.append(np.array([x1, y1], np.int32))
#                         rx.append(np.array([x2, y2], np.int32))
#         if len(rx) == 0:
#             y_right, x_right, y_middle, x_middle, y_left, x_left, r1, m1, l1 = polyfit_lane_left(
#                 lx, erosion_half_resize)
#             left = 1
#         else:
#             y_right, x_right, y_middle, x_middle, y_left, x_left, r1, m1, l1 = polyfit_lane_right(
#                 rx, erosion_half_resize)
#             left = 0
#         if flag==1:
#             if isinstance(b, np.ndarray) and b.size == 0:
#                 if comparison(g, m1) and comparison(r, r1):
#                     transformed_name = 'left turn 2 lanes' + \
#                         str(img_count + 1) + '.png'
#                     image_name = 'left turn 2 lanes ' + str(img_count) + '.png'
#                     return
#                 else:
#                     if comparison(r, m1):
#                         rx = m1
#                         mx = None
#                         lx = None
#                         transformed_name = 'left turn 1 lanes' + \
#                             str(img_count + 1) + '.png'
#                         image_name = 'left turn 1 lanes and one garbage on left' + \
#                             str(img_count) + '.png'
#                     elif comparison(r, l1):
#                         rx = l1
#                         mx = None
#                         lx = None
#                         transformed_name = 'right turn 2 lanes' + \
#                             str(img_count + 1) + '.png'
#                         image_name = 'right turn 2 lanes and one garbage on left' + \
#                             str(img_count) + '.png'
#             else:
#                 if comparison(g, m1) and comparison(b, l1):
#                     transformed_name = 'right turn 2 lanes' + \
#                         str(img_count + 1) + '.png'
#                     image_name = 'right turn 2 lanes ' + str(img_count) + '.png'
#                     return
#                 else:
#                     if comparison(b, m1):
#                         lx = m1
#                         mx = None
#                         rx = None
#                         transformed_name = 'right turn 1 lanes' + \
#                             str(img_count + 1) + '.png'
#                         image_name = 'right turn 1 lanes and one garbage on right' + \
#                             str(img_count) + '.png'
#                     elif comparison(b, r1):
#                         lx = r1
#                         mx = None
#                         rx = None
#                         transformed_name = 'left turn 2 lanes' + \
#                             str(img_count + 1) + '.png'
#                         image_name = 'left turn 2 lanes and one garbage on right' + \
#                             str(img_count) + '.png'

#             if len(rx) == 0:
#                 y_right, x_right, y_middle, x_middle, y_left, x_left, r1, m1, l1 = polyfit_lane_left(
#                     lx, erosion_half_resize)
#                 left = 1
#             else:
#                 y_right, x_right, y_middle, x_middle, y_left, x_left, r1, m1, l1 = polyfit_lane_right(
#                     rx, erosion_half_resize)
#                 left = 0

#     else:
#         for x in range(0, len(lines)):
#             for x1, y1, x2, y2 in lines[x]:
#                 pts = np.array([[x1, y1], [x2, y2]], np.int32)
#                 # cv2.circle(image, [x1, y1 ], 2, (0,255,0),-1)
#                 k = int((x1+x2)/2)
#                 # f=(y1+y2)/2
#                 line_points.append(np.array([x1, y1], np.int32))
#                 line_points.append(np.array([x2, y2], np.int32))

#         # binary_image = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
#         # white_pixel_counts = count_white_pixels(erosion,mid)

#         # weighted_avg = sum(count * x for count, x in zip(white_pixel_counts, mid)) / sum(white_pixel_counts)

#         # print("Weighted average of white pixels:", weighted_avg)

#         # height, width = erosion_half_resize.shape[:2]
#         # if((weighted_avg<width/2)) :
#         if flag==1:
#             if isinstance(b, np.ndarray) and b.size == 0:
#                 rx = line_points
#                 y_right, x_right, y_middle, x_middle, y_left, x_left, r1, m1, l1 = polyfit_lane_right(
#                     rx, erosion_half_resize)
#                 transformed_name = 'left turn' + str(img_count + 1) + '.png'
#                 image_name = 'left turn ' + str(img_count) + '.png'
#             else:
#                 lx = line_points
#                 y_right, x_right, y_middle, x_middle, y_left, x_left, r1, m1, l1 = polyfit_lane_left(
#                     lx, erosion_half_resize)
#                 transformed_name = 'right turn' + str(img_count + 1) + '.png'
#                 image_name = 'right turn ' + str(img_count) + '.png'

#     # Initialize an array of zeros with the same shape as the image
#     reconstructed_image = np.zeros((480, 640), dtype=np.uint8)

#     # Set the specified points to 255
#     for point in mx:
#         reconstructed_image[point[1], point[0]] = 255

#     for point in lx:
#         reconstructed_image[point[1], point[0]] = 255

#     for point in rx:
#         reconstructed_image[point[1], point[0]] = 255

#     polyfitimage = np.zeros((480, 640, 3), dtype=np.uint8)  # 480,640,3
#     for i in range(len(x_left)):
#         polyfitimage[int(x_left[i]), int(y_left[i])] = (0, 0, 255)
#     for i in range(len(x_middle)):
#         polyfitimage[int(x_middle[i]), int(y_middle[i])] = (0, 255, 0)
#     for i in range(len(x_right)):
#         polyfitimage[int(x_right[i]), int(y_right[i])] = (255, 0, 0)

#     if flag == 1:
#         image_path = folder_path+image_name
#         transformed_path = folder_path+transformed_name
#         cv2.imwrite(image_path, erosion_half_resize)
#         cv2.imwrite(transformed_path, transformed_img)
#     prev_img = polyfitimage
#     flag = 1
#     polyfitimage_msg = bridge.cv2_to_imgmsg(
#         np.array(polyfitimage), encoding="bgr8")
#     lane_pub_image.publish(polyfitimage_msg)
#     erosion_msg = bridge.cv2_to_imgmsg(
#         np.array(erosion_half_resize), encoding="8UC1")
#     erosion_pub_image.publish(erosion_msg)
#     # Publish lane coordinates
#     lane_msg = LaneCoordinates()
#     lane_msg.lx = y_right  # wrong have to change
#     lane_msg.mx = y_middle
#     lane_msg.rx = y_left
#     lane_msg.ly = x_right
#     lane_msg.my = x_middle
#     lane_msg.ry = x_left

#     lane_pub.publish(lane_msg)
#     print(j)
#     print(left)
#     j += 1


# def main():
#     rospy.init_node('lane_detector')
#     global bridge, lane_pub, lx, mx, rx, lane_pub_image, erosion_pub_image, ly, my, ry
#     # global stopline_pub
#     bridge = CvBridge()
#     # image_s=rospy.Subscriber('/camera/color/image_raw', Image,call)
#     # Define publisher for lane coordinates
#     lane_pub = rospy.Publisher(
#         '/lane_coordinates', LaneCoordinates, queue_size=10)
#     imu_sub = message_filters.Subscriber('/imu/data', Imu)
#     encoder_sub = rospy.Subscriber('/odo', Float64, encoder_callback)

#     img_sub = message_filters.Subscriber('/camera/color/image_raw', Image)

#     time_interval = 0.1

#     ts = message_filters.ApproximateTimeSynchronizer([imu_sub, img_sub], 20, time_interval,
#                                                      allow_headerless=False)
#     ts.registerCallback(callback)

#     lane_pub_image = rospy.Publisher("/lane_image", Image, queue_size=10)
#     # stopline_pub = rospy.Publisher("/stopline_coord", Int64MultiArray, queue_size=10)
#     erosion_pub_image = rospy.Publisher(
#         "/erosion_lane_image", Image, queue_size=10)
#     lx, mx, rx, ly, my, ry = [], [], [], [], [], []

#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         print("Shutting down")
#     cv2.destroyAllWindows()


# if __name__ == '_main_':
#     main()
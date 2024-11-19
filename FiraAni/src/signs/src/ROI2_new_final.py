#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid


def array_callback(msg):
    global inputa
    inputa = msg.data

def image_callback(msg):

    global image_data
    bridge = CvBridge()
    image_data = bridge.imgmsg_to_cv2(msg, "bgr8")

def main():
    global image_data
    original_image = 
    rospy.init_node('image_processing_node')
    inputa = []
    a=0
    image_data  = []
    rospy.Subscriber("/occupancy_grid_signs", OccupancyGrid, array_callback)
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    # rospy.spin() 
    # while len(image_data) == 0 or len(inputa) == 0:
    #     print("Stuck")
    #     continue
    print("checkpoint1")
    bridge = CvBridge()
    intrinsic_params = np.array([[646.35693359375, 0, 645.4840698242188],
                                   [0, 645.5247192382812, 367.06915283203125],
                                   [0, 0, 1]])
    
    coeffs = [-0.05302947387099266, 0.06136186420917511, 0.00021476151596289128, 0.001250683912076056, -0.01992097683250904]
    print(type(inputa))
    # inputa = inputa / 100.0
    inputa = [x / 100.0 for x in inputa] 
    pt = []
    num_rows = len(inputa) // 3
    for i in range(num_rows):
        row = inputa[i * 3 : (i + 1) * 3]
        pt.append(row)

    pixel_pt = []
    for p in pt:
        point = np.array(p)
        print("Entered")
        x = point[0] / point[2]
        y = point[1] / point[2]
        r2 = x * x + y * y

        f = 1 + coeffs[0] * r2 + coeffs[1] * r2 * r2 + coeffs[4] * r2 * r2 * r2
        x *= f
        y *= f

        dx = x + 2 * coeffs[2] * x * y + coeffs[3] * (r2 + 2 * x * x)
        dy = y + 2 * coeffs[3] * x * y + coeffs[2] * (r2 + 2 * y * y)
        x = dx
        y = dy

        u = x * intrinsic_params[0, 0] + intrinsic_params[0, 2]
        v = y * intrinsic_params[1, 1] + intrinsic_params[1, 2]

        pixel_pt.append([u, v])

    
    image_height, image_width = original_image.shape[:2]

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    pixel_pt = np.array(pixel_pt)
    min_u, max_u = np.min(pixel_pt[:, 0]), np.max(pixel_pt[:, 0])
    min_v, max_v = np.min(pixel_pt[:, 1]), np.max(pixel_pt[:, 1])
    print(pixel_pt)
    min_u, max_u = int(min_u), int(max_u)
    min_v, max_v = int(min_v), int(max_v)

    # for u, v in pixel_pt:
    #     if 0 <= u < image_width and 0 <= v < image_height and min_u <= u <= max_u and min_v <= v <= max_v:
    #         mask[int(v), int(u)] = 255  # Set pixel intensity to 255 (white) at (u, v) coordinates in mask
    mask[min_v:max_v,min_u:max_u]=255
    highlighted_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    

    processed_image_msg = bridge.cv2_to_imgmsg(highlighted_image, encoding='bgr8')
    processed_image_pub = rospy.Publisher("/processed_image_topic", Image, queue_size=10)
    processed_image_pub.publish(processed_image_msg)
    cc
    rospy.loginfo("Processed image published!")
    rospy.spin()


if __name__ == "__main__":
    main()

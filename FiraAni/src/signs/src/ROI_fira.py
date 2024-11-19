#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid

a = 340
inputa = np.array([])
image_data = None

def array_callback(msg):
    global inputa
    inputa = msg.data

def image_callback(msg):
    global image_data
    image_data = msg

def main():
    global inputa, image_data
    global a

    rospy.init_node('image_processing_node')
    rospy.Subscriber("/occupancy_grid_signs", OccupancyGrid, array_callback)
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    processed_image_pub = rospy.Publisher("/processed_image_topic", Image, queue_size=10)
    # inputa=inputa/100
    bridge = CvBridge()

    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        inputa=np.array(inputa)
        inputa=inputa/100
        if inputa.size > 0 and image_data is not None:
            try:
                original_image = bridge.imgmsg_to_cv2(image_data, desired_encoding='bgr8')
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: {0}".format(e))
                continue

            intrinsic_params = np.array([[646.35693359375, 0, 645.4840698242188],
                                         [0, 645.5247192382812, 367.06915283203125],
                                         [0, 0, 1]])
            # intrinsic_params = np.array([[303.7754, 0, 305.9769],
            #                           [0, 302.5833, 288.5363],
            #                           [0, 0, 1]])
            coeffs = [-0.05302947387099266, 0.06136186420917511, 0.00021476151596289128, 0.001250683912076056, -0.01992097683250904]
            
            num_points = inputa.size // 3
            points = inputa.reshape((num_points, 3))
            # print(points)
            pixel_pt = []
            for point in points:
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
            # P = np.hstack((intrinsic_params, np.zeros((3, 1))))
            # P = np.vstack((P, [0, 0, 0, 1]))  
            # for p in points:
            #     p_homogeneous = np.hstack((p, 1))
            #     pixel_coords = np.dot(intrinsic_params, p)
            #     u = pixel_coords[0] / pixel_coords[2]
            #     v = pixel_coords[1] / pixel_coords[2]
            #     pixel_pt.append([u, v])
            print(pixel_pt)

            image_height, image_width = original_image.shape[:2]

            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            pixel_pt = np.array(pixel_pt)

            min_u, max_u = np.clip([np.min(pixel_pt[:, 0]), np.max(pixel_pt[:, 0])], 0, image_width - 1).astype(int)
            min_v, max_v = np.clip([np.min(pixel_pt[:, 1]), np.max(pixel_pt[:, 1])], 0, image_height - 1).astype(int)

            mask[min_v:max_v, min_u:max_u] = 255
            highlighted_image = cv2.bitwise_and(original_image, original_image, mask=mask)
            cv2.imshow(f"No_entry{a}",highlighted_image)
            # cv2.imwrite(f"/home/umic/sign_images/No_entry/No_entry{a}.png",highlighted_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            a=a+1
            try:
                processed_image_msg = bridge.cv2_to_imgmsg(highlighted_image, encoding='bgr8')
                processed_image_pub.publish(processed_image_msg)
                rospy.loginfo("Processed image published!")
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: {0}".format(e))

        rate.sleep()

if __name__ == "__main__":
    main()

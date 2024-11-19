#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import rospy
# from std_msgs.msg import Float64MultiArray
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from cv_only.msg import LaneCoordinates  

# class IPM():
#     def __init__(self):
#         rospy.init_node('goal_mark_node')
#         self.lanes_sub = rospy.Subscriber("/inverse_rgb_topic", Image, self.callback)
#         self.depth_image_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.callback2)
#         self.ipm_lanes_pub = rospy.Publisher("/lane_coordinates", LaneCoordinates, queue_size=10)
#         # self.rgb_ipm_pub=rospy.Publisher("/inverse_rgb_topic", Image, queue_size=10)
#         self.bridge = CvBridge()
#         self.depth=None


#     def callback2(self,msg):
#         rospy.loginfo(f"Image encoding: {msg.encoding}")
#         self.depth=self.bridge.imgmsg_to_cv2(msg,desired_encoding='passthrough')    
#     def callback(self, msg):
#         rgb_img=self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
#         # (w, h) = (640, 480)
#         # # ymax = 480
#         # # x1 = w // 2 - 300
#         # # x2 = w // 2 + 175
#         # # l = 400
#         # # tl = (180, 320)
#         # # bl = (10, 460)
#         # # tr = (400, 320)
#         # # br = (500, 460)
#         # ymax = 480
#         # x1 = w//2 - 300
#         # x2 = w//2 + 175   
#         # l = 400
#         # tl=(180,320)
#         # bl=(66,475)
#         # tr=(400,320)
#         # br=(458,475)
#         # print(rgb_img.shape)
#         b, g, r = cv2.split(rgb_img)
#         pointsl = np.where(b > 0)
#         pointsm=np.where(g>0)
#         pointsr=np.where(r>0)
#         # pointsl = np.column_stack((pointsl[1], pointsl[0], np.ones(pointsl[0].shape)))
#         # pointsm = np.column_stack((pointsm[1], pointsm[0],np.ones(pointsl[0].shape)))
#         # pointsr = np.column_stack((pointsr[1], pointsr[0],np.ones(pointsl[0].shape)))
#         k=np.array([
#             [303.7754,0,305.9769],
#             [0,302.5833,288.5363],
#             [0,0,1]
#             ])
#         print(self.depth.shape)
#         # rospy.loginfo("shape",shape)
#         D=[-0.05302947387099266, 0.06136186420917511, 0.00021476151596289128, 0.001250683912076056, -0.01992097683250904]
#         lanemsg=LaneCoordinates()
#         k_inv = np.linalg.inv(k)
#         for point in pointsr:
#             # if(point[1]>=720 or point[0]>=1280 or self.depth.size == 0):
#             #     break
#             # print(point[1])
#             d=self.depth[point[1],point[0]]
#             d *=0.001
#             # pointc=np.dot(k_inv,point)
#             x=(point[0]-305.9769)/303.7754
#             y=(point[1]-288.5363)/302.5833
#             pointc=[x,y,1.0]
#             # pointc[0]=x
#             # pointc[1]=y
#             # pointc[2]=1
#             r2  = pointc[0]*pointc[0] + pointc[1]*pointc[1];
#             f = 1 + D[0]*r2 + D[1]*r2*r2 + D[4]*r2*r2*r2;
#             ux = pointc[0]*f + 2*D[2]*pointc[0]*pointc[1] + D[3]*(r2 + 2*pointc[0]*pointc[0]);
#             uy = pointc[1]*f + 2*D[3]*pointc[0]*pointc[1] + D[2]*(r2 + 2*pointc[1]*pointc[1]);
#             # pointc[0] = ux;
#             # pointc[1] = uy;
#             # pointc *=d
#             pointc[0] = ux*d;
#             pointc[1] = uy*d;
#             pointc [2]=d
#             pointc[0]=pointc[0]*100 +150
#             pointc[2]=pointc[2]*100 +150
#             lanemsg.rx.append(pointc[0])
#             lanemsg.ry.append(pointc[2])

#         for point in pointsm:
#             # if(point[1]>=720 or point[0]>=1280 or self.depth.size == 0):
#             #     break
#             d=self.depth[point[1],point[0]]
#             d *=0.001
#             # pointc=np.dot(k_inv,point)
#             x=(point[0]-305.9769)/303.7754
#             y=(point[1]-288.5363)/302.5833
#             pointc=[x,y,1.0]
#             # pointc[0]=x
#             # pointc[1]=y
#             # pointc[2]=1
#             r2  = pointc[0]*pointc[0] + pointc[1]*pointc[1];
#             f = 1 + D[0]*r2 + D[1]*r2*r2 + D[4]*r2*r2*r2;
#             ux = pointc[0]*f + 2*D[2]*pointc[0]*pointc[1] + D[3]*(r2 + 2*pointc[0]*pointc[0]);
#             uy = pointc[1]*f + 2*D[3]*pointc[0]*pointc[1] + D[2]*(r2 + 2*pointc[1]*pointc[1]);
#             pointc[0] = ux*d;
#             pointc[1] = uy*d;
#             pointc [2]=d
#             pointc[0]=pointc[0]*100 +150
#             pointc[2]=pointc[2]*100 +150
#             lanemsg.mx.append(pointc[0])
#             lanemsg.my.append(pointc[2])

#         for point in pointsl:
#             # if(point[1]>=720 or point[0]>=1280 or self.depth.size == 0):
#             #     break
#             d=self.depth[point[1],point[0]]
#             d *=0.001
#             # pointc=np.dot(k_inv,point)
#             x=(point[0]-305.9769)/303.7754
#             y=(point[1]-288.5363)/302.5833
#             pointc=[x,y,1.0]
#             # pointc[0]=x
#             # pointc[1]=y
#             # pointc[2]=1
#             r2  = pointc[0]*pointc[0] + pointc[1]*pointc[1];
#             f = 1 + D[0]*r2 + D[1]*r2*r2 + D[4]*r2*r2*r2;
#             ux = pointc[0]*f + 2*D[2]*pointc[0]*pointc[1] + D[3]*(r2 + 2*pointc[0]*pointc[0]);
#             uy = pointc[1]*f + 2*D[3]*pointc[0]*pointc[1] + D[2]*(r2 + 2*pointc[1]*pointc[1]);
#             # pointc[0] = ux;
#             # pointc[1] = uy;
#             # pointc *=d
#             pointc[0] = ux*d;
#             pointc[1] = uy*d;
#             pointc [2]=d
#             pointc[0]=pointc[0]*100 +150
#             pointc[2]=pointc[2]*100 +150
#             lanemsg.lx.append(pointc[0])
#             lanemsg.ly.append(pointc[2])
#         # rospy.loginfo("lx", lx)
#         self.ipm_lanes_pub.publish(lanemsg)

# if __name__ == "__main__":
#     obj = IPM()
#     rospy.spin()

# import rospy
# from std_msgs.msg import Float64MultiArray
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from cv_only.msg import LaneCoordinates  

# class IPM():
#     def __init__(self):
#         rospy.init_node('goal_mark_node')
#         self.lanes_sub = rospy.Subscriber("/inverse_rgb_topic", Image, self.callback)
#         self.depth_image_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.callback2)
#         self.ipm_lanes_pub = rospy.Publisher("/lane_coordinates", LaneCoordinates, queue_size=10)
#         self.bridge = CvBridge()
#         self.depth = None

#     def callback2(self, msg):
#         rospy.loginfo(f"Image encoding: {msg.encoding}")
#         self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#         if self.depth is not None:
#             rospy.loginfo(f"Depth image size: {self.depth.shape}")

#     def callback(self, msg):
#         rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#         rospy.loginfo(f"RGB image size: {rgb_img.shape}")

#         if self.depth is None:
#             rospy.logwarn("Depth data not received yet.")
#             return

#         if self.depth.shape[:2] != rgb_img.shape[:2]:
#             rospy.logerr("Depth and RGB image sizes do not match.")
#             return

#         b, g, r = cv2.split(rgb_img)
#         pointsl = np.where(b > 0)
#         pointsm = np.where(g > 0)
#         pointsr = np.where(r > 0)

#         k = np.array([
#             [303.7754, 0, 305.9769],
#             [0, 302.5833, 288.5363],
#             [0, 0, 1]
#         ])

#         D = [-0.05302947387099266, 0.06136186420917511, 0.00021476151596289128, 0.001250683912076056, -0.01992097683250904]
#         lanemsg = LaneCoordinates()

#         def process_points(points, rx, ry):
#             for y, x in zip(points[0], points[1]):
#                 if y < 0 or y >= self.depth.shape[0] or x < 0 or x >= self.depth.shape[1]:
#                     rospy.logwarn(f"Skipping out-of-bounds point: ({y}, {x})")
#                     continue
#                 d = self.depth[y, x]
#                 d *= 0.001
#                 x_norm = (x - 305.9769) / 303.7754
#                 y_norm = (y - 288.5363) / 302.5833
#                 pointc = [x_norm, y_norm, 1.0]
#                 r2 = pointc[0]**2 + pointc[1]**2
#                 f = 1 + D[0]*r2 + D[1]*r2**2 + D[4]*r2**3
#                 ux = pointc[0]*f + 2*D[2]*pointc[0]*pointc[1] + D[3]*(r2 + 2*pointc[0]**2)
#                 uy = pointc[1]*f + 2*D[3]*pointc[0]*pointc[1] + D[2]*(r2 + 2*pointc[1]**2)
#                 pointc[0] = ux * d
#                 pointc[1] = uy * d
#                 pointc[2] = d
#                 pointc[0] = pointc[0] * 100 + 150
#                 pointc[2] = pointc[2] * 100 + 150
#                 rx.append(pointc[0])
#                 ry.append(pointc[2])

#         process_points(pointsr, lanemsg.rx, lanemsg.ry)
#         process_points(pointsm, lanemsg.mx, lanemsg.my)
#         process_points(pointsl, lanemsg.lx, lanemsg.ly)

#         self.ipm_lanes_pub.publish(lanemsg)

# if __name__ == "__main__":
#     obj = IPM()
#     rospy.spin()


import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_only.msg import LaneCoordinates  

class IPM():
    def __init__(self):
        rospy.init_node('goal_mark_node')
        self.lanes_sub = rospy.Subscriber("/inverse_rgb_topic", Image, self.callback)
        self.depth_image_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.callback2)
        self.ipm_lanes_pub = rospy.Publisher("/lane_coordinates", LaneCoordinates, queue_size=10)
        self.bridge = CvBridge()
        self.depth = None

    def callback2(self, msg):
        rospy.loginfo(f"Image encoding: {msg.encoding}")
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')    

    def callback(self, msg):
        rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        b, g, r = cv2.split(rgb_img)
        pointsl = np.where(b > 0)
        pointsm = np.where(g > 0)
        pointsr = np.where(r > 0)
        
        k = np.array([
            [303.7754, 0, 305.9769],
            [0, 302.5833, 288.5363],
            [0, 0, 1]
        ])
        D = [-0.05302947387099266, 0.06136186420917511, 0.00021476151596289128, 0.001250683912076056, -0.01992097683250904]
        lanemsg = LaneCoordinates()
        
        k_inv = np.linalg.inv(k)
        
        for y, x in zip(pointsl[0], pointsl[1]):
            if y >= 720 or x >= 1280 or self.depth is None:
                continue
            d = self.depth[y, x]
            d *= 0.001
            x_norm = (x - 305.9769) / 303.7754
            y_norm = (y - 288.5363) / 302.5833
            pointc = [x_norm, y_norm, 1.0]
            r2 = x_norm * x_norm + y_norm * y_norm
            f = 1 + D[0] * r2 + D[1] * r2 * r2 + D[4] * r2 * r2 * r2
            ux = x_norm * f + 2 * D[2] * x_norm * y_norm + D[3] * (r2 + 2 * x_norm * x_norm)
            uy = y_norm * f + 2 * D[3] * x_norm * y_norm + D[2] * (r2 + 2 * y_norm * y_norm)
            pointc[0] = ux * d
            pointc[1] = uy * d
            pointc[2] = d
            pointc[0] = pointc[0] * 100 + 150
            pointc[2] = pointc[2] * 100 + 150
            lanemsg.lx.append(int(pointc[0]))
            lanemsg.ly.append(int(pointc[2]))

        for y, x in zip(pointsm[0], pointsm[1]):
            if y >= 720 or x >= 1280 or self.depth is None:
                continue
            d = self.depth[y, x]
            d *= 0.001
            x_norm = (x - 305.9769) / 303.7754
            y_norm = (y - 288.5363) / 302.5833
            pointc = [x_norm, y_norm, 1.0]
            r2 = x_norm * x_norm + y_norm * y_norm
            f = 1 + D[0] * r2 + D[1] * r2 * r2 + D[4] * r2 * r2 * r2
            ux = x_norm * f + 2 * D[2] * x_norm * y_norm + D[3] * (r2 + 2 * x_norm * x_norm)
            uy = y_norm * f + 2 * D[3] * x_norm * y_norm + D[2] * (r2 + 2 * y_norm * y_norm)
            pointc[0] = ux * d
            pointc[1] = uy * d
            pointc[2] = d
            pointc[0] = pointc[0] * 100 + 150
            pointc[2] = pointc[2] * 100 + 150
            lanemsg.mx.append(int(pointc[0]))
            lanemsg.my.append(int(pointc[2]))

        for y, x in zip(pointsr[0], pointsr[1]):
            if y >= 720 or x >= 1280 or self.depth is None:
                continue
            d = self.depth[y, x]
            d *= 0.001
            x_norm = (x - 305.9769) / 303.7754
            y_norm = (y - 288.5363) / 302.5833
            pointc = [x_norm, y_norm, 1.0]
            r2 = x_norm * x_norm + y_norm * y_norm
            f = 1 + D[0] * r2 + D[1] * r2 * r2 + D[4] * r2 * r2 * r2
            ux = x_norm * f + 2 * D[2] * x_norm * y_norm + D[3] * (r2 + 2 * x_norm * x_norm)
            uy = y_norm * f + 2 * D[3] * x_norm * y_norm + D[2] * (r2 + 2 * y_norm * y_norm)
            pointc[0] = ux * d
            pointc[1] = uy * d
            pointc[2] = d
            pointc[0] = pointc[0] * 100 + 150
            pointc[2] = pointc[2] * 100 + 150
            lanemsg.rx.append(int(pointc[0]))
            lanemsg.ry.append(int(pointc[2]))

        lanemsg.lx=lanemsg.lx[::-1]
        lanemsg.rx=lanemsg.rx[::-1]
        lanemsg.mx=lanemsg.mx[::-1]
        lanemsg.my=lanemsg.my[::-1]
        lanemsg.ly=lanemsg.ly[::-1]
        lanemsg.ry=lanemsg.ry[::-1]
        self.ipm_lanes_pub.publish(lanemsg)

if __name__ == "__main__":
    obj = IPM()
    rospy.spin()
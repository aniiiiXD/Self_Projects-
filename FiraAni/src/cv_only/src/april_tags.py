#!/usr/bin/env python3

import rospy
import cv2
import apriltag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
from std_msgs.msg import String


class AprilTagDetector:
    def __init__(self):
        rospy.init_node('apriltag_detector', anonymous=False)
        self.bridge = CvBridge()
        self.depth_img = None

        self.crop_pub = rospy.Publisher("/crop_image", Image, queue_size=5)
        self.predict_pub = rospy.Publisher("/ml_image", String, queue_size=5)

        self.image_sub = rospy.Subscriber(
            '/camera/color/image_raw', Image, self.img_cbk)
        self.depth_sub = rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image, self.depth_cbk)

        self.min_bound = rospy.get_param("/LOW_BOUND", 100)
        self.max_bound = rospy.get_param("/UP_BOUND", 1000)

        self.detector = apriltag.Detector()
        rospy.loginfo("AprilTag detector node initialized.")

    def img_cbk(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        local_depth_img = self.depth_img

        # SEEMS TO WORK BETTER WITH BINARY THRESHOLD !!!!!!!!!!!!!!!!!!
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        results = self.detector.detect(gray)
        points = []
        ids = []

        for r in results:
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            cv2.line(cv_image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(cv_image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(cv_image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(cv_image, ptD, ptA, (0, 255, 0), 2)
            cv2.putText(cv_image, str(r.tag_id), (ptA[0], ptA[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)

            cX, cY = int(r.center[0]), int(r.center[1])
            curr_points = [ (cX, cY), ptA, ptB, ptC, ptD ]
            points.append(curr_points)
            ids.append(r.tag_id)

            self.crop_pub.publish(self.bridge.cv2_to_imgmsg(cv_image))

        for i in range(len(ids)):
            id_msg = String()
            id = ids[i]
            name = 'Empty'
            curr_pts = points[i]

            if self.range_check(curr_pts, local_depth_img):
                if id == 0:
                    name = "no_entry"
                    id_msg.data = name
                    self.predict_pub.publish(id_msg)
                    rospy.loginfo(name)
                elif id == 1:
                    name = "dead_end"
                    id_msg.data = name
                    self.predict_pub.publish(id_msg)
                    rospy.loginfo(name)
                elif id == 2:
                    name = "right"
                    id_msg.data = name
                    self.predict_pub.publish(id_msg)
                    rospy.loginfo(name)
                elif id == 3:
                    name = "left"
                    id_msg.data = name
                    self.predict_pub.publish(id_msg)
                    rospy.loginfo(name)
                elif id == 4:
                    name = "forward"
                    id_msg.data = name
                    self.predict_pub.publish(id_msg)
                    rospy.loginfo(name)
                elif id == 5:
                    name = "stop"
                    id_msg.data = name
                    self.predict_pub.publish(id_msg)
                    rospy.loginfo(name)

    def depth_cbk(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg)
        # self.depth_points_counter(self.depth_img)

    def range_check(self, pts, depth_img):
        depth_pts_arr = (pts[0],
                         ((pts[0][0] + pts[1][0]) // 2,
                          (pts[0][1] + pts[1][1]) // 2),
                         ((pts[0][0] + pts[2][0]) // 2,
                          (pts[0][1] + pts[2][1]) // 2),
                         ((pts[0][0] + pts[3][0]) // 2,
                          (pts[0][1] + pts[3][1]) // 2),
                         ((pts[0][0] + pts[4][0]) // 2,
                          (pts[0][1] + pts[4][1]) // 2)
                         )
        avg_depth = 0
        num_pts = 0

        for pt in depth_pts_arr:
            depth = depth_img[pt[1], pt[0]]
            if depth:
                avg_depth = (avg_depth * num_pts + depth) / (num_pts + 1)
                num_pts += 1

        print("DEPTH: ", avg_depth)

        if avg_depth > self.min_bound and avg_depth < self.max_bound:
            return True
        else:
            return False


if __name__ == '__main__':
    AprilTagDetector()
    rospy.spin()


# 0 - No entry
# 1 - Dead end
# 2 - Right
# 3 - Left
# 4 - Forward
# 5 - Stop

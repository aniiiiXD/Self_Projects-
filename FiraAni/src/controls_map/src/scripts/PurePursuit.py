#! /usr/bin/python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64
import matplotlib.pyplot as plt
import time

class Controller:
    def __init__(self):
        self.q = rospy.get_param('/controls_node/q')
        self.m = rospy.get_param('/controls_node/m')
        self.t_clip_min = rospy.get_param('/controls_node/t_clip_min')
        self.t_clip_max = rospy.get_param('/controls_node/t_clip_max')
        self.wheelbase = rospy.get_param('/controls_node/wheelbase')
        self.loop_rate = rospy.get_param('/controls_node/loop_rate')
        self.agression = rospy.get_param('/controls_node/agression')
        self.steer_bound = rospy.get_param('/controls_node/steer_bound')
        self.wps = []
        self.x = None 
        self.y = None
        self.v = None
        self.w = None
        self.odom = None

        self.velocity_pub = rospy.Publisher('/velocity', Float64, queue_size=10)
        self.steer_pub = rospy.Publisher('/steer', Float64, queue_size=10)
        rospy.Subscriber('/odom',Float64,self.odom_cb)
        rospy.Subscriber('/best_trajectory', PoseArray, self.wp_cb)

    def odom_cb(self,data):
        self.odom = data.data

    def wp_cb(self, data):
        self.px = []
        self.py = []
        self.wps = []
        for i in range(len(data.poses)):
            self.x = data.poses[i].position.x
            self.y = data.poses[i].position.y
            self.v = data.poses[i].position.z
            self.w = data.poses[i].orientation.w
            self.px.append(self.x)
            self.py.append(self.y)
            self.wps.append([self.x, self.y, self.v, self.w])

    def control_loop(self):
        rate = rospy.Rate(self.loop_rate)
        
        while not rospy.is_shutdown():
            if(self.odom==None):
                continue
            if (len(self.wps)==0):
                continue
            print(self.odom)

            v_target = self.wps[0][2]
            lookahead_distance = self.m * v_target + self.q
            lookahead_distance = np.clip(lookahead_distance, self.t_clip_min, self.t_clip_max)
            print("ld",lookahead_distance)
            lookahead_point = self.lookaheadpoint(lookahead_distance)
            print("x",lookahead_point[0]," y",lookahead_point[1])
            steering_angle = self.get_actuation(lookahead_point)
            steering_angle = np.clip(np.rad2deg(steering_angle),-self.steer_bound,self.steer_bound)*self.agression
            if(v_target - self.odom ==0.05):
                steering_angle = 0 
            print("velocity", v_target)
            print("steer",steering_angle)             
            # self.velocity_pub.publish(0.1)
            self.velocity_pub.publish(0.2)
            self.steer_pub.publish(steering_angle)
            print(len(self.wps))
            # plt.plot(self.px,self.py)
            # plt.scatter(lookahead_point[0],lookahead_point[1])
            # plt.show()
            print(time.time())
            rate.sleep()

    def distance(self, arrA, arrB):
        return (((arrA[0] - arrB[0]) ** 2) + ((arrA[1] - arrB[1]) ** 2)) ** 0.5

    def lookaheadpoint(self, distance):
        dist = 0
        i = 0

        while dist < distance:
            i = (i + 1) % len(self.wps)
            dist = self.distance([self.wps[i][0], self.wps[i][1]], [self.wps[0][0], self.wps[0][1]])
        return [self.wps[i][0], self.wps[i][1]]

    def get_actuation(self, lookahead_point):
        waypoint_y = lookahead_point[1]
        if np.abs(waypoint_y) < 1e-6:
            return 0
        radius = (0.21/np.tan(29) + np.linalg.norm(lookahead_point)) ** 2 / (2.0 * waypoint_y)
        steering_angle = np.arctan(self.wheelbase / radius)
        return steering_angle

if __name__ == "__main__":
    rospy.init_node('controller_node')
    controller = Controller()
    controller.control_loop()

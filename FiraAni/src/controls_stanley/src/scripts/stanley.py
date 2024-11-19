#!/usr/bin/python3

import numpy as np 
import rospy
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64

class StanleyController:
    def __init__(self):

        self.k = rospy.get_param('/controls_node_stanley/k')
        self.k_soft = rospy.get_param('/controls_node_stanley/k_soft')
        self.loop_rate = rospy.get_param('/controls_node_stanley/loop_rate')
        self.steer_bound = rospy.get_param('/controls_node_stanley/steer_bound')
        
        self.wps = []
        self.x = None 
        self.y = None
        self.v = None
        self.w = None

        self.velocity_pub = rospy.Publisher('/velocity', Float64, queue_size=10)
        self.steer_pub = rospy.Publisher('/steer', Float64, queue_size=10)
    
        rospy.Subscriber('/best_trajectory', PoseArray, self.wp_cb)

    def wp_cb(self, data):
        self.wps = []
        for i in range(len(data.poses)):
            self.x = data.poses[i].position.x
            self.y = data.poses[i].position.y
            self.v = data.poses[i].position.z
            self.w = data.poses[i].orientation.w
            self.wps.append([self.x, self.y, self.v, self.w])

    def control_loop(self):
        rate = rospy.Rate(self.loop_rate)
        while not rospy.is_shutdown():
            if (len(self.wps)==0):
                continue
            self.cte = self.wps[-1][1]
            self.v_target = self.wps[0][2] 
            self.steering_angle = np.arctan(self.k*self.cte / (self.k_soft + self.v_target))
            steering_angle = np.clip(np.rad2deg(self.steering_angle), -self.steer_bound, self.steer_bound)
            print(steering_angle)
            self.velocity_pub.publish(self.v_target)
            self.steer_pub.publish(steering_angle)
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node('controller_node')
    controller = StanleyController()
    controller.control_loop()
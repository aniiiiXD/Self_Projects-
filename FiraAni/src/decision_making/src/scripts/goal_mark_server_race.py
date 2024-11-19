#!/usr/bin/env python3

import rospy
import time
import numpy as np
from matplotlib import pyplot as plt
import tf.transformations as tft
from decision_making.msg import LaneCoordinates ,race
from geometry_msgs.msg import PoseStamped, Quaternion , Point
from std_msgs.msg import Bool , String 

class DM_Node:

    def __init__(self):
        self.lanes_array_topic = "/lane_coordinates"    
        self.local_goal_topic = "/local_goal_dm"   
        self.local_goal_vis_topic = "/local_goal_dm_vis"
        self.vel = "/best_trajectory"   
        self.race_state = "/race_states"
        self.midpoint = "/midpoint"
      

        self.lx = []
        self.mx = []
        self.rx = []
        self.ly = []
        self.my = []
        self.ry = []
        self.flag = 0 
        self.mid_x = None 
        self.mid_y = None 
        

        self.dist = 200
        self.index = -400
        self.loop_rate = 10 
        self.time = None 

        self.local_goal = PoseStamped()
        self.local_goal_vis = PoseStamped()

        self.in_left_lane = False
        self.obstacle_detected = False
        self.middle_lane_bool = False 
        self.obstacle_lane = None 

        rospy.Subscriber(self.lanes_array_topic, LaneCoordinates, self.lanes_callback)
        rospy.Subscriber("/middle_lane_bool", Bool , self.middle_lane_cb)
        rospy.Subscriber("/states" , Bool , self.states_callback)
        rospy.Subscriber(self.race_state , Bool , self.race_callback)
        rospy.Subscriber(self.midpoint , Point , self.mid_callback)
        self.local_goal_pub = rospy.Publisher(self.local_goal_topic, PoseStamped, queue_size=1)
        self.local_goal_vis_pub = rospy.Publisher(self.local_goal_vis_topic, PoseStamped, queue_size=1)


    def race_callback(self , msg):
        self.in_left_lane = not msg.lane
        self.obstacle_detected = msg.obstacle
        self.obstacle_lane = msg.obstacle_lane 

    def middle_lane_cb(self,msg):
        self.middle_lane_bool = msg.data

    def lanes_callback(self, msg):
        self.rx = np.array(msg.lx[::-1]) + 480 
        self.mx = np.array(msg.mx[::-1]) + 480
        self.rx = np.array(msg.rx[::-1]) + 480
        self.ly = -np.array(msg.ly[::-1]) + 1600
        self.my = -np.array(msg.my[::-1]) + 1600
        self.ry = -np.array(msg.ry[::-1]) + 1600
        self.flag = 1 

    def mid_callback(self,msg):
        self.x = msg.x 
        self.y = msg.y 
        


    def markLocalGoal(self, x_array, y_array, distance):
        self.x = x_array[self.index]
        self.y = y_array[self.index]

        self.dx = (x_array[self.index+1] - x_array[self.index-1]) / 2
        self.dy = (y_array[self.index+1] - y_array[self.index-1]) / 2
        
        self.slope_of_normal = - (self.dx / self.dy)

        if self.slope_of_normal >= 0:
            self.orientation = abs(np.arctan(self.slope_of_normal))
            self.x_goal = self.x - distance * np.cos(self.orientation)
            self.y_goal = self.y - distance * np.sin(self.orientation)
            self.yaw = self.orientation 
        else:
            self.orientation = abs(np.arctan(self.slope_of_normal))
            self.x_goal = self.x - distance * np.cos(self.orientation)
            self.y_goal = self.y + distance * np.sin(self.orientation)
            self.yaw = -self.orientation

        return self.x_goal, self.y_goal, self.yaw
    
    def publishLocalGoal(self, x, y, yaw):
        self.local_goal.header.stamp = rospy.Time.now()
        self.local_goal.header.frame_id = 'camera_color_optical_frame'
        self.local_goal.pose.position.x = (y - 800) / 800
        self.local_goal.pose.position.y = -(x - 800) / 800
        self.local_goal.pose.position.z = 0
        self.quat = tft.quaternion_from_euler(0, 0, yaw)
        self.local_goal_orientation = Quaternion()
        self.local_goal_orientation.x = self.quat[0]
        self.local_goal_orientation.y = self.quat[1]
        self.local_goal_orientation.z = self.quat[2]
        self.local_goal_orientation.w = self.quat[3]
        self.local_goal.pose.orientation = self.local_goal_orientation
        self.local_goal_pub.publish(self.local_goal)

        print(self.local_goal.pose.position.x, self.local_goal.pose.position.y, yaw)

        self.local_goal_vis.header.stamp = rospy.Time.now()
        self.local_goal_vis.header.frame_id = 'camera_color_optical_frame'
        self.local_goal_vis.pose.position.x = x / 800
        self.local_goal_vis.pose.position.y = y / 800
        self.local_goal_vis.pose.position.z = 0
        self.quat_vis = tft.quaternion_from_euler(0, 0, np.pi / 2 + yaw)
        self.local_goal_vis_orientation = Quaternion()
        self.local_goal_vis_orientation.x = self.quat_vis[0]
        self.local_goal_vis_orientation.y = self.quat_vis[1]
        self.local_goal_vis_orientation.z = self.quat_vis[2]
        self.local_goal_vis_orientation.w = self.quat_vis[3]
        self.local_goal_vis.pose.orientation = self.local_goal_vis_orientation
        self.local_goal_vis_pub.publish(self.local_goal_vis)

    def run_algo(self):
        while self.flag == 0 and not rospy.is_shutdown():
            continue

        rate = rospy.Rate(self.loop_rate)

        while not rospy.is_shutdown():
            self.start_time = time.time()

            if not self.in_left_lane:
                if not self.obstacle_detected:
                    rospy.loginfo("Path clear!, keep Right")
                    max_len = np.max([len(self.ly), len(self.my), len(self.ry)])
                    if max_len == len(self.ry):
                        self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.rx, self.ry, self.dist)
                    elif max_len == len(self.my):
                        self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.mx, self.my, -self.dist)
                    else:
                        self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.lx, self.ly, -3 * self.dist)

                else:
                    if self.obstacle_lane:
                        if self.obstacle_lane:
                            rospy.loginfo("Obstacle Detected! Change lane to Left")
                            max_len = np.max([len(self.ly), len(self.my), len(self.ry)])
                            if max_len == len(self.ry):
                                self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.rx, self.ry, 3 * self.dist)
                            elif max_len == len(self.my):
                                self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.mx, self.my, self.dist)
                            else:
                                self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.lx, self.ly, -self.dist) 
                        else:
                            rospy.loginfo("Path clear! Keep Right")
                            max_len = np.max([len(self.ly), len(self.my), len(self.ry)])
                            if max_len == len(self.ry):
                                self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.rx, self.ry, self.dist)
                            elif max_len == len(self.my):
                                self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.mx, self.my, -self.dist)
                            else:
                                self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.lx, self.ly, -3 * self.dist)
            

            else:
                if not self.obstacle_detected:
                    rospy.loginfo("Right lane clear! , re-change")
                    max_len = np.max([len(self.ly), len(self.my), len(self.ry)])
                    if max_len == len(self.ry):
                        self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.rx, self.ry, self.dist)
                    elif max_len == len(self.my):
                        self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.mx, self.my, -self.dist)
                    else:
                        self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.lx, self.ly, -3 * self.dist)
                
                else :
                    if self.obstacle_lane:
                        rospy.loginfo("keep_left")
                        max_len = np.max([len(self.ly), len(self.my), len(self.ry)])
                        if max_len == len(self.ry):
                            self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.rx, self.ry, 3 * self.dist)
                        elif max_len == len(self.my):
                            self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.mx, self.my, self.dist)
                        else:
                            self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.lx, self.ly, -self.dist) 

                    else:
                        rospy.loginfo("Right lane clear! , re-change")
                        max_len = np.max([len(self.ly), len(self.my), len(self.ry)])
                        if max_len == len(self.ry):
                            self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.rx, self.ry, self.dist)
                        elif max_len == len(self.my):
                            self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.mx, self.my, -self.dist)
                        else:
                            self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.lx, self.ly, -3 * self.dist)


            if abs(self.mid_y-self.localgoal_y) <= 50 : 
                self.localgoal_y += 25 

            self.publishLocalGoal(self.localgoal_x, self.localgoal_y, self.local_yaw)
            self.end_time = time.time()
            rospy.loginfo(self.end_time - self.start_time)

            rate.sleep()

if __name__ == "__main__":
    rospy.init_node('dm_node', anonymous=True)
    dmnode = DM_Node()
    dmnode.run_algo()

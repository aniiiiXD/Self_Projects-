#!/usr/bin/env python3

import rospy
import time
import numpy as np
from matplotlib import pyplot as plt
import tf.transformations as tft
from decision_making.msg import LaneCoordinates , states 
from std_msgs.msg import Bool ,Float64
from geometry_msgs.msg import PoseStamped, Quaternion , PoseArray , Point 
from message_filters import ApproximateTimeSynchronizer

class DM_Node:

    def __init__(self):
        self.lanes_array_topic = "/lane_coordinates"    
        self.local_goal_topic = "/local_goal_dm"   
        self.local_goal_vis_topic = "/local_goal_dm_vis"    
        self.states_topic = "/states"
        self.midpoint = "/midpoint"
        self.obs_init_time=-1.0

        self.lx = []
        self.mx = []
        self.rx = []
        self.ly = []
        self.my = []
        self.ry = []
        self.mid_x = None 
        self.mid_y = None 
        self.flag = 0 


        # self.dist = rospy.get_param("dm_node/distance")
        self.dist = 100
        self.loop_rate =rospy.get_param("dm_node/loop_rate")

        self.local_goal = PoseStamped()
        self.local_goal_vis = PoseStamped()

        self.in_left_lane = None
        # self.obstacle_detected = False
        self.obstacle_detected = None
        self.counter = []
        self.delay = 0
        self.middle_bool = None
        self.yaw = None
        self.yaw_flag = 0 
        self.tan_pub = rospy.Publisher("/tan", Float64, queue_size=10)
       
        self.tan_pub = rospy.Publisher("/tan", Float64, queue_size=10)
        self.lanes_sub = rospy.Subscriber(self.lanes_array_topic, LaneCoordinates, self.lanes_callback)
        self.middle_lane_sub = rospy.Subscriber("/middle_lane" ,Bool , self.middle_callback )
        self.states_sub= rospy.Subscriber(self.states_topic, states , self.states_callback)
        self.local_goal_pub = rospy.Publisher(self.local_goal_topic, PoseStamped, queue_size=1)
        self.local_goal_vis_pub = rospy.Publisher(self.local_goal_vis_topic, PoseStamped, queue_size=1)
        rospy.Subscriber(self.midpoint,Point , self.mid_callback )
        rospy.Subscriber("/global_yaw", Float64, self.yaw_callback)

    def yaw_callback(self, msg):
        self.yaw = int(msg.data * 1000)
        if self.obstacle_detected :
            self.counter.append(self.yaw)
            rospy.loginfo(f"{len(self.counter)} , {self.counter[-1]},{ self.counter[0]}")
            if self.yaw >= self.counter[0] - 0.05*self.yaw and self.yaw <= self.counter[0] + self.yaw*0.5  :
                self.yaw_flag = 1 

    def lanes_callback(self, msg):
        self.lx = np.array(msg.lx) - 415 
        # self.mx = np.array(msg.mx) - 415 
        self.rx = np.array(msg.rx) - 415 
        # self.ly = -np.array(msg.ly[::-1]) + 1600
        # self.my = -np.array(msg.my[::-1]) + 1600
        # self.ry = -np.array(msg.ry[::-1]) + 1600

        self.ly = np.array(msg.ly[::-1]) 
        # self.my = np.array(msg.my[::-1]) 
        self.ry = np.array(msg.ry[::-1])    
        # self.ly = (-np.array(msg.ly) +480)/(0.008/0.0010625) + 500/2 + (23*1.773-20)/(100*0.008) 
        # self.my = (-np.array(msg.my) +480)/(0.008/0.0010625) + 500/2 + (23*1.773-20)/(100*0.008)
        # self.ry = (-np.array(msg.ry) +480)/(0.008/0.0010625) + 500/2 + (23*1.773-20)/(100*0.008) 
        self.flag = 1 


    def middle_callback(self,msg):
        # self.in_left_lane = not msg.lane_state
        # if msg.obstacle_state:
        #     self.obstacle_detected = True
        #     self.obs_init_time = time.time()
        result = msg.data
        self.middle_bool = result 
    
    def states_callback(self , msg):
        # self.in_left_lane = not msg.lane_state
        # # while(msg.obstacle_state != None):
        # #     # print(msg.obstacle_state)
        # #     self.obstacle_detected = msg.obstacle_state
        # if msg.obstacle_stat
        #     self.obstacle_detected = True\        #     self.obs_init_time = time.time()

        self.in_left_lane = not msg.lane_state
        # while(msg.obstacle_state != None):
        #     # print(msg.obstacle_state)
        #     self.obstacle_detected = msg.obstacle_state
        self.obstacle_detected = msg.obstacle_state

    def mid_callback(self,msg):
        x = msg.x 
        y = msg.y

        self.mid_x = x 
        self.mid_y = y 

    def markLocalGoal(self, x_array, y_array, distance):      
        if len(x_array) == 0 and len(y_array) == 0:
            return None, None, None 
        
        try : 
            self.index = 4*len(y_array)//5
            if(self.index)>len(y_array):
                self.index = -1  
            # elif self.middle_bool:
            #     self.index = 2*len(y_array)//5
            elif not self.in_left_lane and self.obstacle_detected:
                self.index = 2*len(y_array)//5
            elif self.in_left_lane and not self.obstacle_detected:
                self.index = -1
        except IndexError : 
            rospy.logerr("lane length not sufficient ")


        if len(self.rx) == 0 or len(self.lx) == 0 :
            self.dist = 89 
        else:
            print(abs(self.lx[0]-self.rx[0])//4)
            self.dist = abs(self.lx[0]-self.rx[0])//4

        
        print(self.dist)

        self.x = x_array[self.index]
        self.y = y_array[self.index]
        print(self.index)

        self.dx = (x_array[self.index+2] - x_array[self.index-2]) / 2
        self.dy = (y_array[self.index+1] - y_array[self.index-1]) / 2
        
        self.slope_of_normal = - (self.dx / self.dy)
        
        self.tan_pub.publish(self.slope_of_normal)

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
        self.local_goal.pose.position.y = - x * 0.00140625
        self.local_goal.pose.position.x = y * 0.0010625 + 0.2
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
        self.local_goal_vis.pose.position.x = x * 0.00140625 + 62.5 * 0.016    
        self.local_goal_vis.pose.position.y = y * 0.0010625 + 0.2 + 62.5 * 0.016
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
        self.start_time = time.time()
        while self.flag == 0:
            continue

        rate = rospy.Rate(self.loop_rate)
        
        if not self.in_left_lane:
            if not self.obstacle_detected:
                rospy.loginfo("Path clear! Keep Right")
                max_len = np.max([len(self.ly), len(self.my), len(self.ry)])
                print(max_len)
                if max_len == 0 :
                    return 
                elif max_len == len(self.ry):
                    self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.rx, self.ry, self.dist)
                elif max_len == len(self.my):
                    self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.mx, self.my, -self.dist)
                else:
                    self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.lx, self.ly, -3 * self.dist)
                self.delay = 1 
            else:
                rospy.loginfo("Obstacle Detected! Change lane to Left")
                max_len = np.max([len(self.ly), len(self.my), len(self.ry)])
                if max_len == 0 :
                    return
                elif max_len == len(self.ry):
                    self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.rx, self.ry, 3 * self.dist)
                elif max_len == len(self.my):
                    self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.mx, self.my, self.dist)
                else:
                    self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.lx, self.ly, -self.dist)
                self.delay = 1 
        else:
            if self.obstacle_detected:
                rospy.loginfo("Obstacle detected! Keep Left")
                max_len = np.max([len(self.ly), len(self.my), len(self.ry)])
                if max_len == 0 :
                    return
                elif max_len == len(self.ry):
                    self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.rx, self.ry, 3 * self.dist)
                elif max_len == len(self.my):
                    self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.mx, self.my, self.dist)
                else:
                    self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.lx, self.ly, -self.dist)
                self.delay = 1 
            else:
                print("THIS IS YAW FLAGGGGGGGGG" , self.yaw_flag )
                if self.yaw_flag == 1 : 
                    start = time.time()
                    while(time.time()-start <= 1.7 and False):
                        rospy.loginfo("Yaw Scam activated!!!! WorKKKKKKKKKKKKKKKKK")
                        print(time.time()-start)
                        max_len = np.max([len(self.ly), len(self.my), len(self.ry)])
                        if max_len == 0 :
                            return
                        elif max_len == len(self.ry):
                            self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.rx, self.ry, 3 * self.dist)
                        elif max_len == len(self.my):
                            self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.mx, self.my, self.dist)
                        else:
                            self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.lx, self.ly, -self.dist)
                        self.counter = [] 
                        self.yaw_flag = 0 
                        
                


                rospy.loginfo("Obstacle crossed! Go back to Right")
                max_len = np.max([len(self.ly), len(self.my), len(self.ry)])
                if max_len == 0 :
                    return
                elif max_len == len(self.ry):
                    self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.rx, self.ry, self.dist)
                elif max_len == len(self.my):
                    self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.mx, self.my, -self.dist)
                else:
                    self.localgoal_x, self.localgoal_y, self.local_yaw = self.markLocalGoal(self.lx, self.ly, -3 * self.dist)
                self.delay+=1 

        if self.localgoal_x is None or self.localgoal_y is None or self.local_yaw is None:
            return

        if self.in_left_lane and not self.obstacle_detected :
            self.publishLocalGoal(self.localgoal_x-20, self.localgoal_y+100, self.local_yaw)
        else:    
            self.publishLocalGoal(self.localgoal_x, self.localgoal_y, self.local_yaw)
        self.obstacle_detected = False  # NEW LINE
        self.end_time = time.time()

        # if self.obstacle_detected and (self.end_time - self.obs_init_time > 1):
        #     self.obstacle_detected = False
        # else:
        #     if self.end_time - self.obs_init_time < 1:
        #         print("     ",self.end_time - self.obs_init_time)

        # plt.plot(self.lx,self.ly)
        # plt.plot(self.rx,self.ry)
        # plt.scatter(self.localgoal_x, self.localgoal_y)
        # plt.show()

        rospy.loginfo(self.end_time - self.start_time)

        rate.sleep()

if __name__ == "__main__":
    rospy.init_node('dm_node', anonymous=True)
    dmnode = DM_Node()
    while not rospy.is_shutdown():
        try:
            dmnode.run_algo()
        except IndexError:
            rospy.logerr("lane length not sufficient")

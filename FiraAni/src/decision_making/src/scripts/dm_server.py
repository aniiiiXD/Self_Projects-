#!/usr/bin/env python3

import numpy as np
import math 
from nav_msgs.msg import OccupancyGrid
import rospy 
import time
from decision_making.msg import LaneCoordinates , states
from geometry_msgs.msg import Point
from std_msgs.msg import String, Int64MultiArray, Bool , Int64 , Float64


class LaneChange:
    def __init__(self):
        self.local_states_topic = "/states"                 # Publisher 
        self.sign_topic= "/ml_image"                       # Subscriber 
        self.zebra_topic = "/zebra"                         # Publisherx``
        # self.output_grid_topic = "/output_grid"             # Publisher   
        self.middle_lane_bool = "/middle_lane"         # Publisher
        self.stopline_topic = "/stopline_flag"    # Publisher
        self.occupancy_grid_topic = "/occupancy_grid"       # Subscriber
        self.lanes_topic = "/lane_coordinates"
        self.midpoint  = "/midpoint"                        # Subscriber 
        self.turn_radius = "/turn_rad"                      # Publisher 
     
        self.in_left_lane = None 
        self.obstacle_detected = False
        self.obstacle_lane_bool = True 
        self.lane_change = None 
        self.nothing_flag = None 
        self.car_pose_x = 800 
        self.car_pose_y = 800
        self.radius = 90
        self.result = None 
        self.sign = None 
        self.sample_size = 7
        self.turn_rad_value = None 
        self.stoppie = None 

        # self.in_left_lane = False 
        # self.obstacle_detected = False 

        self.counter_array =  []
        self.current_task = None 
        self.test_list = []
        self.ult_list = []

        self.sign_array = []
        self.lx = []
        self.ly = []
        self.mx = []
        self.my = [] 
        self.rx = []
        self.ry = []
        self.flag = 0 
        self.stop_flag = None 
        self.stop_list = []
        # state = states()

        rospy.Subscriber(self.zebra_topic, Int64MultiArray, self.zebra_callback)
        rospy.Subscriber(self.sign_topic, String, self.sign_callback)
        self.local_states_topic = rospy.Publisher(self.local_states_topic, states, queue_size=20)
        self.middle_lane_bool = rospy.Publisher(self.middle_lane_bool, Bool, queue_size=10)
        rospy.Subscriber(self.stopline_topic, Bool , self.stopline_callback)
        rospy.Subscriber(self.lanes_topic,LaneCoordinates , self.lanes_callback)
        rospy.Subscriber("/midpoint" , Point , self.midpoint_callback)
        rospy.Subscriber("/nothing" , Bool , self.nothing_call)
        self.turn_radius_pub = rospy.Publisher("/turn_rad",Float64, queue_size=10 )

    def nothing_call(self,msg):
        self.nothing_flag = msg.data

    def stopline_callback(self,msg):
    
        self.stoppie = msg.data
        # print(self.stoppie)
        if self.stoppie:
            rospy.loginfo("publishing one")
            self.turn_rad_value = 1 
            self.turn_radius_pub.publish(1)
            rospy.loginfo("sleep activated for 15 secs")
            rospy.sleep(15)
        else:
            rospy.loginfo("publishing zero")
            self.turn_radius_pub.publish(0)
        # self.stop_list.append(self.stoppie)
        # while(len(self.stop_list) >= 2):
        #     if self.stop_list[-1] != self.stop_list[-2]:
        #         self.execute = True
        #     else:
        #         self.execute = False
    

    def mod_signs(self, array):
        
        frequency_dict = {}
        max_frequency = 0
        most_frequent = []

        for string in array:
            if string in frequency_dict:
                frequency_dict[string] += 1
            else:
                frequency_dict[string] = 1
        
            if frequency_dict[string] > max_frequency:
                max_frequency = frequency_dict[string]
                most_frequent = [string]
            elif frequency_dict[string] == max_frequency:
                most_frequent.append(string)
        
        return most_frequent

    def sign_callback(self, msg):
        self.result = msg.data 
        # if self.result != "Wrong sign detected":
        self.sign_array.append(msg.data)

        # if len(self.sign_array) >= self.sample_size :
        #     for i in range(self.sample_size):
        #         self.test_list.append(self.sign_array(i))
        #         # counter = sum(1 for sign in self.test_list if sign == "left")

        #         # for i in range(self.sample_size):
        #         #     self.ult_list = []
        #         #     if self.test_list[i] != "left":
        #         #         self.ult_list.append(self.test_list[i])
        #         #     else : 
        #         #         counter += 1 
        #         # print(self.test_list)
        #         # if counter > self.sample_size -2: 
        #         #     print(counter)
        #         #     self.sign = "left"
        #         # else:
        #         #     sign = self.mod_signs(self.sign_array)
        #         #     if sign == "deadend" or sign == "no entry":
        #         #             for i in range(self.sample_size):
        #         #                 if self.sign_array[i] == "right" or self.sign_array[i] == "forward" or self.sign_array[i]=="left": 
        #         #                     sign = self.sign_array[i]
        #         #                     break 
        #         #                 # else:
        #         #                 #     self.sign = "left"occ

        #     sign = self.mod_signs(self.test_list)
        #     if sign == "dead_end" or sign == "no_entry":
        #         for j in range(len(self.test_list)):
        #             if self.test_list[j] != "dead_end" or self.test_list[j] != "no_entry" :
        #                 sign = self.test_list[i] 
        #                 break
        # sign = self.mod_signs(self.sign_array[:self.sample_size])
        # print(sign)
        # if sign == 'left':
        #     self.turn_radius_pub.publish(1.0)
        # elif sign == ['right']:
        #     print("going in condition")occ
        #     self.turn_radius_pub.publish(5.0)
        # elif sign == "forward":
        #     self.turn_radius_pub.publish(7.0)
        # elif sign == "stop":
        #     self.turn_radius_pub.publish(2.0)
        # elif sign == "left":
        #     self.turn_radius_pub.publish(1.0)

    def zebra_callback(self, msg):
        zebra_data = msg.data[1]
        if zebra_data > 400: 
           pass 

    def sign_calc(self):

        # if self.obstacle_detected:
        #     self.sign = None 
        
        # else :
        # print("start")
        # while len(self.sign_array) == self.sample_size :
        # print("initialised")
        # for i in range(self.sample_size):
        #     self.test_list.append(self.sign_array(i))
        # counter = sum(1 for sign in self.test_list if sign == "left")

        # for i in range(self.sample_size):
        #     self.ult_list = []
        #     if self.test_list[i] != "left":
        #         self.ult_list.append(self.test_list[i])
        #     else : 
        #         counter += 1 
        # print(self.test_list)
        # if counter > self.sample_size -2: 
        #     print(counter)
        #     self.sign = "left"
        # else:
        if self.nothing_flag:
            sign = ['Wrong sign detected']
        else :
            sign = self.mod_signs(self.sign_array)
            if sign == ['deadend']or sign == ['no entry']:
                    for i in range(self.sample_size):
                        if self.sign_array[i] == ['right'] or self.sign_array[i] == ['forward'] or self.sign_array[i]==["left"]: 
                            sign = self.sign_array[i]
                            break 
                        # else:
                        #     self.sign = "left"
            # else:
            #     sign = self.mod_signs(self.sign_array)

            # print("before called")
            # print(sign)
            self.publish_sign(sign)
            # print(self.ult_list)
            self.sign_array = self.sign_array[:self.sample_size]
            # self.sign_array=[]
            return sign 

    def publish_sign(self , sign):
        if sign == ['forward']:
            print("DM getting Forward")
            self.turn_rad_value = 7.0
            self.turn_radius_pub.publish(7.0)
        elif  sign== ['right']:
            print("DM getting right")
            self.turn_rad_value = 6.0
            self.turn_radius_pub.publish(6.0)
        elif sign == ['left']:
            print("DM getting left")
            self.turn_rad_value = 2.0
            self.turn_radius_pub.publish(2.0)
        elif sign == ['deadend']:
            print("DM getting deadend")
            self.turn_radius_pub.publish(0.45)
        elif  sign== ['no_entry']:
            print("DM getting no_entry")
            self.turn_radius_pub.publish()  # Not publishing any value here
        elif sign == ['stop']:
            print("ruk jaa bhai")
            self.turn_radius_pub.publish(8.0)
        # else:
        #     self.turn_radius_pub.publish(0)
        # print("sign callback working")

    def zebra_callback(self, msg):
        zebra_data = msg.data[1]
        if zebra_data > 400:
            pass



    def lanes_callback(self, msg):
        self.lx=np.array(msg.lx[::-1])-415    # Notation is very bad, it needs to be fixed 
        #self.mx=np.array(msg.mx[::-1])+480 # print(self.sign_array)
        self.rx=np.array(msg.rx[::-1])-415
        self.ly=np.array(msg.ly[::-1])
        #self.my=np.array(msg.my[::-1])
        self.ry=np.array(msg.ry[::-1])
        # self.lx = np.array(msg.lx) 
        # self.mx = np.array(msg.mx) 
        # self.rx = np.array(msg.rx) 
        # self.ly = np.array(msg.ly) 
        # self.my = np.array(msg.my) 
        # self.ry = np.array(msg.ry) 
        self.flag = 1 

    def find_nearest_lane(self , mid_x, mid_y, lx, ly, rx, ry):
        default_dist = 87*2 

        if (len(lx)!=0 and len(ly)!=0 and len(rx)!=0 and len(ry)!=0):
            y_n_l = min(range(len(ly)), key=lambda i: abs(ly[i] - mid_y))
            point_l = ly[y_n_l]
            pxl = lx[y_n_l]
    
            r_n_l = min(range(len(ry)), key=lambda i: abs(ry[i] - mid_y))
            point_r = ry[r_n_l]
            pxr = rx[r_n_l]
    
            left_distance = abs(pxl-mid_x)
            right_distance = abs(pxr-mid_x)
            if left_distance <= right_distance:
                return False  
            else:
                return True 

        elif len(rx)== 0 and len(ly)!= 0 : 
            y_n_l = min(range(len(ly)), key=lambda i: abs(ly[i] - mid_y))
            point_l = ly[y_n_l]
            pxl = lx[y_n_l]

            left_distance = abs(pxl-mid_x)
            if left_distance > default_dist:
                return True
            else:
                return False 
                

        elif len(lx) == 0 and len(rx) != 0 : 
            r_n_l = min(range(len(ry)), key=lambda i: abs(ry[i] - mid_y))
            point_r = ry[r_n_l]
            pxr = rx[r_n_l]
    
            right_distance = abs(pxr-mid_x)
            if right_distance > default_dist:
                return False
            else:
                return True 
            
        else:
            return True
        
        #rospy.loginfo(left_distance , right_distance) 

    def midpoint_callback(self,msg):
        x = msg.x 
        y = msg.y 
        z = msg.z
        # print (x , y, z)

        if(x != 0 or y!= 0 ):
            # pose_x = x 
            # if 
            self.obstacle_detected = True
    
            self.obstacle_lane_bool = self.find_nearest_lane(x, y, self.lx, self.ly, self.rx, self.ry)
            rospy.loginfo(self.obstacle_lane_bool)
        else:
            self.obstacle_detected = False

        # self.counter_array.append(self.obstacle_detected)
        # print("midpoint callback")


    def calc_states(self):
        rospy.Rate(5)
        state = states()
        start_time = time.time()
                
        # if len(self.mx) != 0:
        #     middle_lane_x = self.mx 
        #     if middle_lane_x[0] < 800:
        #         state.lane_state = True 
        #     elif middle_lane_x[0] >800:
        #         state.lane_state = False 

        # if len(self.mx) == 0 :
        #     if self.obstacle_detected : 
        #         state.lane_state = False 
        #     else : 
        #         state.lane_state = True 

        if len(self.lx)!= 0 and len(self.rx) != 0 :
            diff1 = abs(self.lx[0]-80)
            diff2 = abs(self.rx[0]-80)
            print(diff1,diff2)
            if diff1 > diff2 : 
                state.lane_state = True 
            elif diff2 > diff1 : 
                state.lane_state = False 
        else: 
            
            state.lane_state = False


        state.obstacle_state =  self.obstacle_detected
        state.obstacle_lane = self.obstacle_lane_bool
        # print(self.counter_array)
      

        a = state.lane_state 
        b = state.obstacle_state 
        c = state.obstacle_lane

        if a and not b : 
            self.middle_lane_bool.publish(False)
        elif a and b : 
            if c :
                self.middle_lane_bool.publish(True)
            else : 
                self.middle_lane_bool.publish(False) 
        elif not a and b : 
            if c: 
                self.middle_lane_bool.publish(False)
            else : 
                self.middle_lane_bool.publish(True) 
        elif not a and not b : 
            self.middle_lane_bool.publish(True)

        #print(a , b)
        # print(self.sign)
        # print(self.result)
        # print(self.sign_array)
        sign = self.sign_calc()
        # print(len(self.sign_array))
        # print(len(self.test_list))
        # print(len(self.ult_list))
        # print(self.sign_array)
        # print(self.test_list)
        # print(self.ult_list)
        # print(sign)
        # self.publish_sign(sign)
        self.local_states_topic.publish(state)  
        rospy.loginfo(self.turn_rad_value)
        # print(self.execute)
        end_time = time.time()
        # print(end_time-start_time)

    def intersection_call(self):
        arr =self.sign_array
        if (arr[-1] != ""):
            self.turn_radius        

if __name__ == "__main__": 
        try:
            rospy.init_node('states_node')
            lanechange = LaneChange()
            while not rospy.is_shutdown():
                lanechange.calc_states()
        except rospy.ROSInterruptException as e:
            rospy.logerr(f"Interrupt = {e}")
        except Exception as e:
            rospy.logerr(f"exception lol = {e}")
  
        

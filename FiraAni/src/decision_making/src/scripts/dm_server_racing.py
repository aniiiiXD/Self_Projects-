#!/usr/bin/env python3

import numpy as np
import math 
from nav_msgs.msg import OccupancyGrid
import rospy 
import time
from decision_making.msg import LaneCoordinates , race 
from geometry_msgs.msg import Point
from std_msgs.msg import String, Int64MultiArray, Bool , Int64 , Float64


class LaneChange:
    def __init__(self):
        self.race_states_topic = "/races"                 # Publisher 
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
        self.obstacle_lane_bool = None 
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
        self.execute = None 

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

        self.turn_radius_pub = rospy.Publisher("/turn_rad",Float64, queue_size=10 )
        self.race_states_topic = rospy.Publisher(self.race_states_topic, race, queue_size=20)
        self.middle_lane_bool = rospy.Publisher(self.middle_lane_bool, Bool, queue_size=10)
        rospy.Subscriber(self.sign_topic, String, self.sign_callback)
        rospy.Subscriber(self.zebra_topic, Int64MultiArray, self.zebra_callback)
        rospy.Subscriber(self.stopline_topic, Bool , self.stopline_callback)
        rospy.Subscriber(self.lanes_topic,LaneCoordinates , self.lanes_callback)
        rospy.Subscriber("/midpoint" , Point , self.midpoint_callback)
        rospy.Subscriber("/nothing" , Bool , self.nothing_call)

    def nothing_call(self,msg):
        self.nothing_flag = msg.data

    def stopline_callback(self,msg):
        r=rospy.Rate(1)
        self.stoppie = msg.data
        # print(self.stoppie)
        if self.stoppie:
            print("publishing one")
            self.turn_radius_pub.publish(1)
        else:
            print("publishing zero")
            self.turn_radius_pub.publish(0)
        # self.stop_list.append(self.stoppie)
        # while(len(self.stop_list) >= 2):
        #     if self.stop_list[-1] != self.stop_list[-2]:
        #         self.execute = True
        #     else:
        #         self.execute = False
        r.sleep()

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
            self.turn_rad_value = 0.0
            self.turn_radius_pub.publish(0)
        elif  sign== ['right']:
            print("DM getting right")
            self.turn_rad_value = 5.0
            self.turn_radius_pub.publish(5.0)
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
        self.lx=np.array(msg.lx[::-1])+480    # Notation is very bad, it needs to be fixed 
        self.mx=np.array(msg.mx[::-1])+480 # print(self.sign_array)
        self.rx=np.array(msg.rx[::-1])+480
        self.ly=-np.array(msg.ly[::-1])+1600
        self.my=-np.array(msg.my[::-1])+1600
        self.ry=-np.array(msg.ry[::-1])+1600
        # self.lx = np.array(msg.lx) 
        # self.mx = np.array(msg.mx) 
        # self.rx = np.array(msg.rx) 
        # self.ly = np.array(msg.ly) 
        # self.my = np.array(msg.my) 
        # self.ry = np.array(msg.ry) 
        self.flag = 1 

    def find_nearest_y_and_compare_x(mid_x, mid_y, lane_x_array, lane_y_array):

        nearest_y_index = min(range(len(lane_y_array)), key=lambda i: abs(lane_y_array[i] - mid_y))
        nearest_y = lane_y_array[nearest_y_index]

        
        corresponding_x = lane_x_array[nearest_y_index]

    
        if mid_x == corresponding_x:
            comparison_result = "Equal"
        elif mid_x > corresponding_x:
            comparison_result = "Greater"
        else:
            comparison_result = "Lesser"

        return nearest_y, corresponding_x , comparison_result

    def midpoint_callback(self,msg):
        x = msg.x 
        y = msg.y 
        z = msg.z

        

        if(x != 0 or y!= 0 ):
            self.obstacle_detected = True
            nearest_y , corresponding_x ,result= self.find_nearest_y_and_compare_x(x, y, self.mx, self.my)
            if result == "Greater":
                self.obstacle_lane_bool = True 
            elif result == "Lesser":
                self.obstacle_lane_bool = False 
        else:
            self.obstacle_detected = False



    def calc_states(self):
        rospy.Rate(5)
        state = race()
        # print("calc state started ")
        start_time = time.time()
   
        # counter = 1  
        # a = self.counter_array
        # # print(a)
        # while len(a) >= 2  :
        #     if( a[-1] != a[-2]):
        #         pass 
        
                
        if len(self.mx) != 0:
            middle_lane_x = self.mx 
            if middle_lane_x[0] < 800:
                state.lane = True 
            elif middle_lane_x[0] >800:
                state.lane = False 

        if len(self.mx) == 0 :
            if self.obstacle_detected : 
                state.lane = False 
            else : 
                state.lane = True 


        # if self.obstacle_detected == None :
        #     self.obstacle_state = False
        # else:
        #     state.obstacle_state =  self.obstacle_detected
        state.obstacle =  self.obstacle_detected

        # print(self.counter_array)
      

        a = state.lane
        b = state.obstacle
        c = state.obstacle_lane

        if a and not b : 
            self.middle_lane_bool.publish(False)
        elif a and b and c  : 
            self.middle_lane_bool.publish(True)
        elif a and b and not c  : 
            self.middle_lane_bool.publish(False)
        elif not a and b and c: 
            self.middle_lane_bool.publish(False)
        elif not a and b and  not c: 
            self.middle_lane_bool.publish(True)
        elif not a and not b : 
            self.middle_lane_bool.publish(True)

        print(a , b , c)
        # print(self.sign)
        # print(self.result)
        # print(self.sign_array)
        # sign = self.sign_calc()
        # print(len(self.sign_array))
        # print(len(self.test_list))
        # print(len(self.ult_list))
        # print(self.sign_array)
        # print(self.test_list)
        # print(self.ult_list)
        # print(sign)
        # self.publish_sign(sign)
        self.race_states_topic.publish(state)  
        # print(self.turn_rad_value)
        # print(self.execute)
        end_time = time.time()
        # print(end_time-start_time)

    # def intersection_call(self):
    #     arr =self.sign_array
    #     if (arr[-1] != ""):
    #         self.turn_radius        

if __name__ == "__main__": 
        rospy.init_node('states_node')
        lanechange = LaneChange()
        while not rospy.is_shutdown():
            lanechange.calc_states()
             
        

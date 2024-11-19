#!/usr/bin/env python3

import rospy 
import numpy as np 
from std_msgs.msg import Float64 , Int16 , Bool 
from geometry_msgs.msg import PoseStamped 
import time 

class laal_rekha:

    def __init__(self):
        self.current_yaw = None 
        self.prev_error = 0
        self.acc_error = 0
        self.dt = 0.1
        self.prev_time = time.time()

        self.vel_pub = rospy.Publisher("/velocity" , Int16 , queue_size=10)
        self.steer_pub = rospy.Publisher("/steer" , Int16 , queue_size=10)
        rospy.Subscriber("/local_goal_dm", PoseStamped , self.goal_cb )
        # rospy.Subscriber("/filtered_yaw", Float64 , self.yaw_cb)
        

    def goal_cb(self,msg): 
        
        kp = 1
        ki = 0
        kd = 0

        x = msg.pose.position.x
        y = msg.pose.position.y

        vel_init = 0.1

        curr_err = -1 * np.arctan(x/y)

        self.dt = time.time() - self.prev_time

        e_d = (curr_err - self.prev_error) / self.dt

        # print(curr_err , self.dt)

        self.acc_error += curr_err * self.dt
        # print(self.acc_error)
        self.prev_error = np.arctan(y/x)
        # print(self.prev_error)
        steer = kp * curr_err + ki * self.acc_error + kd * e_d 

        self.prev_time = time.time()

        self.vel_pub.publish(vel_init*1000)
        if abs(steer) <=30:
            self.steer_pub.publish(int(steer*1800/np.pi))
        else:
            steer = np.sign(steer)*30
            self.steer_pub.publish(int(steer*1800/np.pi))
        rospy.loginfo(f"{steer} and {vel_init}")


if __name__ == "__main__" : 
    
    rospy.init_node("laal_rekha")
    while not rospy.is_shutdown():
        rekha = laal_rekha()


        
    
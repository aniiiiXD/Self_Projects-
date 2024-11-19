#!/usr/bin/env python3

import rospy 
from std_msgs.msg import Float64, Int16 
from decision_making.msg import states , race 
import matplotlib as plt 

class Scam():

    def __init__(self):
        self.result = None 
        self.obstacle_detected = None
        self.observe_array = []
        self.count_array = []
        self.count = 0 

        self.odom_pub = rospy.Publisher("/final_speed", Int16, queue_size=10)
        rospy.Subscriber("/tan", Float64, self.scam_cb)
        rospy.Subscriber("/states", states , self.state_cb)

        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            rate.sleep()

    def state_cb(self,msg):
        self.obstacle_detected = msg.obstacle_state 


    def scam_cb(self, msg):
        count += 1 
        nm = abs(msg.data)
        self.observe_array.append(nm)
        self.count_array.append(count)

        if len(self.count_array) > 200 : 
            plt.plot(self.count_array, self.observe_array)
            plt.show()
        if nm >= 0 and nm <0.08 :
            self.odom_pub.publish(500)
            print(500)
        elif nm >= 0.087 and nm < 0.1763:
            self.odom_pub.publish(300)
            print(400)
        elif nm >= 0.176 and nm < 0.267:
            self.odom_pub.publish(200)
            print(300)
        elif nm > 0.267 and nm <0.3639: 
            self.odom_pub.publish(200)
            print(200)
        elif nm >= 0.363 and nm< 0.4663: 
            self.odom_pub.publish(150)
            print(150)
        elif nm >= 0.4663: 
            self.odom_pub.publish(100)
            print(100)

        elif self.obstacle_detected:
            self.odom_pub.publish(180)
            print("jaldi wahan se hato ")
	
	

if __name__ == "__main__":
    rospy.init_node("scam_odom")
    Scam()


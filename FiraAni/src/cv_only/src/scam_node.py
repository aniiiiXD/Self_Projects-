#!/usr/bin/env python3

import rospy 
from time import time 
from sensor_msgs.msg import Image 

def image_callback(msg):
	global ctr
	rate = rospy.getParam("/dm_node/loop_rate")
	ctr += 1
	
	result = msg.data 
	if (ctr % 2 == 0 ) : 
	    scam_pub.publish(result)


def main():
	scam_pub = rospy.Publisher("/camera/color/image_raw/throttle" , Image , queue_size=1)
	camera_sub = rospy.Subscriber("/camera/color/image_raw",Image , image_callback) 
	
	rospy.spin()
	
if __name__ == "__main__":

	rospy.init_node("scam")
	global ctr
    	ctr = 0
    	main()
	
	
	

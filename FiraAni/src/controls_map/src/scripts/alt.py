#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32
from std_msgs.msg import Float64

def velocity_callback(msg):
    global velocity
    velocity = msg.data

def steer_callback(msg):
    global steer
    steer = msg.data

def publish_combined_message(event):
    global velocity
    global steer

    # Combine velocity and steer into a single string\
    sign=0
    if(steer>0):
        sign=1
    
#36.8   0.76  1
#368000  076   1000000
    combined_message = abs(steer)*10000+sign*1000000+abs(velocity)*100
    # Publish the combined message
    pub_combined.publish(combined_message)

    rospy.loginfo(f"Published combined message: {combined_message}")

if __name__ == '__main__':
    rospy.init_node('alt_publisher_node')

    # Initialize variables
    velocity = 0.0
    steer = 0.0

    # Setup subscribers
    rospy.Subscriber('/velocity', Float64, velocity_callback)
    rospy.Subscriber('/steer', Float64, steer_callback)

    # Setup publisher
    pub_combined = rospy.Publisher('/alt', Int32, queue_size=1)

    # Setup a timer to publish at 20 Hz
    rate = rospy.Rate(20)  # 20 Hz
    while not rospy.is_shutdown():
        publish_combined_message(None)  # Call function to publish combined message
        rate.sleep()



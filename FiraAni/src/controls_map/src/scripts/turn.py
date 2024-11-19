#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64

def publish_velocity():
    # Initialize the node with the name 'velocity_publisher'
    rospy.init_node('velocity_publisher', anonymous=True)

    # Create a publisher object for the /velocity topic, with Float64 message type
    velocity_publisher = rospy.Publisher('/velocity', Float64, queue_size=10)

    # Set the publish rate (10 Hz)
    rate = rospy.Rate(10)  # 10 Hz

    # Start time
    start_time = rospy.Time.now().to_sec()
    duration = 0.3  # seconds

    while not rospy.is_shutdown():
        current_time = rospy.Time.now().to_sec()
        elapsed_time = current_time - start_time

        # Check if 3 seconds have passed
        if elapsed_time < duration:
            velocity_msg = Float64()
            velocity_msg.data = 0.4  # Example velocity value
            velocity_publisher.publish(velocity_msg)
            rospy.loginfo(f"Publishing velocity: {velocity_msg.data}")
            rate.sleep()
        else:
            velocity_msg.data = -1.0  # Example velocity value
            velocity_publisher.publish(velocity_msg)
            rospy.loginfo(f"Publishing velocity: {velocity_msg.data}")
            rate.sleep()            
            break

    rospy.loginfo("Publishing finished.")

if __name__ == '__main__':
    try:
        publish_velocity()
    except rospy.ROSInterruptException:
        pass


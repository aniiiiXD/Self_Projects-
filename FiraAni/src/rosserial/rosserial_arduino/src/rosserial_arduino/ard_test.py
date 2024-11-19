#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64  # Assuming you're using Int16 for /steer and /velocity messages
import serial
v=0
s=0
def steer_callback(msg):
    # Callback function for /steer topic
    #serial_port.write(f"S{msg.data}\n".encode())  # Assuming "S" denotes steer
    v=msg.data

def velocity_callback(msg):
    # Callback function for /velocity topic
    #serial_port.write(f"V{msg.data}\n".encode())  # Assuming "V" denotes velocity
    s=msg.data


if __name__ == '__main__':
    rospy.init_node('arduino_interface_node', anonymous=True)

    # Serial port configuration
    serial_port = serial.Serial('/dev/ttyUSB0', 9600)  # Adjust port and baudrate as necessary
    v_corr=int(V*100)
    s_corr=int(s*10)
    str_out=str(v_corr)+' '+str(s_corr)+'\n'
    # Subscribers
    serial_port.write(str_out)
    rospy.Subscriber('/steer', Int16, steer_callback)
    rospy.Subscriber('/velocity', Int16, velocity_callback)

    rospy.spin()


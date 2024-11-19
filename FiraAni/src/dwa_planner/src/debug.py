#!/usr/bin/env python3 

import rospy 
from cv_only.msg import LaneCoordinates
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, PoseStamped


class DetectNode:

	def __init__(self):	
		self.grid = None
		self.goal = None
		self.count = 0
		rospy.init_node("mp_detector")
		rospy.Subscriber("/occupancy_grid", OccupancyGrid, self.occ_cb)
		rospy.Subscriber("/local_goal_dm_vis", PoseStamped, self.dm_cb)
		rospy.Subscriber("/best_trajectory", PoseArray, self.mp_cb)
		rospy.spin()
		
	def occ_cb(self,msg):
		self.grid = msg 
	
		
	def dm_cb(self,msg):
		self.goal = msg 
		
	def mp_cb(self,msg):
		for pose in msg.poses:
			if pose.position.x !=0 or pose.position.y !=0 or pose.position.z !=0:
				return
		rospy.loginfo("Empty")
		grid = self.grid
		goal = self.goal
		f = open(f"grid{self.count}.txt", 'w')
		f.write("New Message\n")
		f.write("Goal -> x = " + str(goal.pose.position.x) + " , y = " + str(goal.pose.position.y) + "\n")
		f.write("Grid ->\n" + str(grid.data) + "\n\n\n")
		f.close()
		self.count+=1
			
if __name__ == '__main__':
	DetectNode()
	

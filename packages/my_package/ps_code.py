#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import Range
from duckietown_msgs.msg import WheelsCmdStamped


class StopAtObstacleNode(DTROS):

    def __init__(self, node_name):

        super(StopAtObstacleNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # publisher 
        self.pub = rospy.Publisher(
            f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd",
            WheelsCmdStamped,
            queue_size=1
        )

        # subscriber 
        self.sub = rospy.Subscriber(
            f"/{self._vehicle_name}/front_center_tof_driver_node/range",
            Range,
            self.callback
        )

        rospy.loginfo("Wheel control node started")


    def callback(self, msg):

        distance = msg.range

        cmd = WheelsCmdStamped()

        
        if distance > 0.10:

            cmd.vel_left = 0.3
            cmd.vel_right = 0.3

            rospy.loginfo(f"Distance {distance:.2f} m → Moving forward")

        
        else:

            cmd.vel_left = 0.0
            cmd.vel_right = 0.0

            rospy.loginfo(f"Obstacle detected at {distance:.2f} m → STOP")

        self.pub.publish(cmd)


if __name__ == "__main__":

    print("ps_code.py çalışıyor.")
    node = StopAtObstacleNode(node_name="stop_at_obstacle_node")

    rospy.spin()

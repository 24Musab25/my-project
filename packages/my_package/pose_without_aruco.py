#!/usr/bin/env python3

import rospy
import os
import math
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int8

try:
    from duckietown_msgs.msg import Twist2DStamped
    HAS_DUCKIETOWN_MSGS = True
except ImportError:
    from geometry_msgs.msg import Twist
    HAS_DUCKIETOWN_MSGS = False

class PoseWithoutArucoNode:
    def __init__(self):
        rospy.init_node('pose_without_aruco_node', anonymous=True)
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'duckiebot')

        self.is_aruco_detected = False
        
        # Dead reckoning state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_time = rospy.Time.now()

        # Listen to ArUco detection status
        self.detect_sub = rospy.Subscriber(
            "/duckie_mission/is_aruco_detected",
            Bool,
            self.detect_callback,
            queue_size=1
        )
        
        # Listen to ground-truth pose to sync state when ArUco IS visible
        self.pose_sync_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/estimated_pose",
            PoseStamped,
            self.pose_sync_callback,
            queue_size=5
        )

        # Odometry / Velocity integration
        if HAS_DUCKIETOWN_MSGS:
            self.vel_sub = rospy.Subscriber(
                f"/{self._vehicle_name}/kinematics_node/velocity",
                Twist2DStamped,
                self.velocity_callback,
                queue_size=1
            )
        else:
            self.vel_sub = rospy.Subscriber(
                f"/{self._vehicle_name}/cmd_vel",
                Twist,
                self.velocity_callback_standard,
                queue_size=1
            )

        self.pose_pub = rospy.Publisher(
            f"/{self._vehicle_name}/estimated_pose",
            PoseStamped,
            queue_size=10
        )
        self.mode_pub = rospy.Publisher(
            "/duckie_mission/mission_mode",
            Int8,
            queue_size=1
        )
        rospy.loginfo("Pose Without Aruco (Odometry Fallback) Node Started")
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

    def detect_callback(self, msg):
        self.is_aruco_detected = msg.data

    def pose_sync_callback(self, msg):
        # Sync state ONLY when ArUco is active so we start from the correct spot
        # when we lose it.
        if self.is_aruco_detected:
            self.x = msg.pose.position.x
            self.y = msg.pose.position.y
            q_z = msg.pose.orientation.z
            q_w = msg.pose.orientation.w
            self.theta = 2.0 * math.atan2(q_z, q_w)

    def integrate_velocities(self, v, omega):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # Ignore huge gaps
        if dt > 0.5: 
            dt = 0.0

        self.x += v * dt * math.cos(self.theta)
        self.y += v * dt * math.sin(self.theta)
        self.theta += omega * dt

    def velocity_callback(self, msg):
        self.integrate_velocities(msg.v, msg.omega)

    def velocity_callback_standard(self, msg):
        self.integrate_velocities(msg.linear.x, msg.angular.z)

    def timer_callback(self, event):
        # We only publish pose and mode if ArUco is NOT detected
        if not self.is_aruco_detected:
            self.mode_pub.publish(Int8(2))

            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "world"
            pose_msg.pose.position.x = float(self.x)
            pose_msg.pose.position.y = float(self.y)
            pose_msg.pose.position.z = 0.0
            
            pose_msg.pose.orientation.z = math.sin(self.theta / 2.0)
            pose_msg.pose.orientation.w = math.cos(self.theta / 2.0)

            self.pose_pub.publish(pose_msg)

def main():
    node = PoseWithoutArucoNode()
    rospy.spin()

if __name__ == '__main__':
    main()

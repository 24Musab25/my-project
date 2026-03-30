#!/usr/bin/env python3

import os
import math
import rospy

# Optional DTROS usage if in duckietown. dtros takes node_name and node_type.
try:
    from duckietown.dtros import DTROS, NodeType
    HAS_DTROS = True
except ImportError:
    HAS_DTROS = False

from sensor_msgs.msg import Range
from geometry_msgs.msg import PoseStamped

try:
    from duckietown_msgs.msg import WheelsCmdStamped
    HAS_WHEELS_CMD = True
except ImportError:
    HAS_WHEELS_CMD = False


base_class = DTROS if HAS_DTROS else object

class DuckieMoverNode(base_class):
    def __init__(self):
        if HAS_DTROS:
            super(DuckieMoverNode, self).__init__(node_name="duckie_mover_node", node_type=NodeType.GENERIC)
        else:
            rospy.init_node('duckie_mover_node', anonymous=True)
            
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'duckiebot')

        # Proportional Control Parameters
        self.target_x = 0.0  # Go back to origin 
        self.target_y = 0.0
        self.k_v = 0.8       # Gain for linear velocity based on distance
        self.k_w = 1.0       # Gain for angular velocity based on angle error
        self.v_max = 0.4     # Max speed limit to prevent flying off the track
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0

        self.obstacle_distance = 1.0 # 1 meter default safe 

        if HAS_WHEELS_CMD:
            self.pub = rospy.Publisher(
                f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd",
                WheelsCmdStamped,
                queue_size=1
            )

        self.pose_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/estimated_pose",
            PoseStamped,
            self.pose_callback
        )
        self.tof_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/front_center_tof_driver_node/range",
            Range,
            self.tof_callback
        )
        
        # Control loop at 10Hz
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        rospy.loginfo("Duckie Mover Node Started. P-Control based on remaining distance.")

    def tof_callback(self, msg):
        self.obstacle_distance = msg.range

    def pose_callback(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        q_z = msg.pose.orientation.z
        q_w = msg.pose.orientation.w
        self.current_theta = 2.0 * math.atan2(q_z, q_w)

    def control_loop(self, event):
        if not HAS_WHEELS_CMD:
            return
            
        cmd = WheelsCmdStamped()
        
        # 1. Safety override from TOF
        if self.obstacle_distance <= 0.10:
            cmd.vel_left = 0.0
            cmd.vel_right = 0.0
            self.pub.publish(cmd)
            # rospy.logwarn_throttle(2.0, f"Obstacle Too Close! ({self.obstacle_distance:.2f}m) STOPPING.")
            return

        # 2. Distance and angle calculations
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        remaining_dist = math.sqrt(dx**2 + dy**2)
        
        angle_to_target = math.atan2(dy, dx)
        angle_error = angle_to_target - self.current_theta
        
        # Normalize angle error to [-pi, pi]
        while angle_error > math.pi:  angle_error -= 2 * math.pi
        while angle_error < -math.pi: angle_error += 2 * math.pi

        # Target reached gracefully
        if remaining_dist < 0.1:
            cmd.vel_left = 0.0
            cmd.vel_right = 0.0
            self.pub.publish(cmd)
            return

        # 3. Proportional control laws
        # The further away, the faster v
        # The higher the angle error, the higher omega
        v = self.k_v * remaining_dist
        omega = self.k_w * angle_error
        
        # Apply strict max limits
        v = min(v, self.v_max)
        
        # Basic differential drive mixing (approx. width factor = 0.3 here for quick turn)
        v_l = v - omega * 0.3
        v_r = v + omega * 0.3
        
        # Check if mixing exceeded maximum allowed wheel speed, rescale gracefully
        max_wheel = max(abs(v_l), abs(v_r))
        if max_wheel > self.v_max:
            v_l = (v_l / max_wheel) * self.v_max
            v_r = (v_r / max_wheel) * self.v_max

        cmd.vel_left = float(v_l)
        cmd.vel_right = float(v_r)
        
        # Sends differential drive speeds
        self.pub.publish(cmd)

if __name__ == '__main__':
    node = DuckieMoverNode()
    rospy.spin()

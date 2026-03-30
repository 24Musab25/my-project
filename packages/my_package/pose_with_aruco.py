#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int8

class PoseWithArucoNode:
    def __init__(self):
        rospy.init_node('pose_with_aruco_node', anonymous=True)
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'duckiebot')

        # Camera Intrinsics
        self.K = np.array([
            [332.6544121604296, 0.0, 316.81243765089346],
            [0.0, 329.6706637774356, 239.3524836292827],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.d = np.array([
            -0.3006822939569327, 0.0674782489139842,
            -0.008971789541553121, 0.0014690287922984812, 0.0
        ], dtype=np.float32)

        self.marker_size = 0.05
        self.known_marker_poses = {
            33: (0.0, 1.1, 0.1, 0.0, 0.0, 0.0),
            26: (0.2, 1.1, 0.1, 0.0, 0.0, np.pi / 2),
        }

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.is_aruco_detected = False
        self.origin_world_pose = None

        # Subscribers
        self.image_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/image/compressed",
            CompressedImage,
            self.image_callback,
            queue_size=1
        )
        self.detect_sub = rospy.Subscriber(
            "/duckie_mission/is_aruco_detected",
            Bool,
            self.detect_callback,
            queue_size=1
        )
        
        # Publishers
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

        rospy.loginfo("Pose With Aruco Node Started")
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

    def detect_callback(self, msg):
        self.is_aruco_detected = msg.data

    def timer_callback(self, event):
        if self.is_aruco_detected:
            self.mode_pub.publish(Int8(1))

    def euler_to_matrix(self, roll, pitch, yaw, x, y, z):
        R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        R = R_z @ R_y @ R_x
        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3], T[1, 3], T[2, 3] = x, y, z
        return T

    def image_callback(self, msg):
        # Sadece ArUco algılandığı kesinse ve flag true ise işlem yap. Boş karelerde yorma.
        if not self.is_aruco_detected:
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception:
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.K, self.d)
            
            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                if marker_id in self.known_marker_poses:
                    rvec, tvec = rvecs[i][0], tvecs[i][0]
                    T_world_marker = self.euler_to_matrix(*self.known_marker_poses[marker_id])
                    
                    R_cam_marker, _ = cv2.Rodrigues(rvec)
                    T_cam_marker = np.eye(4)
                    T_cam_marker[:3, :3] = R_cam_marker
                    T_cam_marker[:3, 3] = tvec.reshape(3)

                    T_marker_cam = np.linalg.inv(T_cam_marker)
                    T_world_cam = T_world_marker @ T_marker_cam
                    
                    absolute_world_x = T_world_cam[0, 3]
                    absolute_world_y = T_world_cam[1, 3]

                    if self.origin_world_pose is None:
                        self.origin_world_pose = np.array([absolute_world_x, absolute_world_y], dtype=np.float32)

                    relative_x = absolute_world_x - self.origin_world_pose[0]
                    relative_y = absolute_world_y - self.origin_world_pose[1]

                    # Basit heading hesabı
                    yaw = np.arctan2(T_world_cam[1, 0], T_world_cam[0, 0])

                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = rospy.Time.now()
                    pose_msg.header.frame_id = "world"
                    pose_msg.pose.position.x = float(relative_x)
                    pose_msg.pose.position.y = float(relative_y)
                    pose_msg.pose.position.z = 0.0
                    pose_msg.pose.orientation.z = np.sin(yaw / 2.0)
                    pose_msg.pose.orientation.w = np.cos(yaw / 2.0)

                    self.pose_pub.publish(pose_msg)
                    break 

def main():
    node = PoseWithArucoNode()
    rospy.spin()

if __name__ == '__main__':
    main()

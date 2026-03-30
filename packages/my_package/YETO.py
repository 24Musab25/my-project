#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped, Twist
from cv_bridge import CvBridge


class ArUcoLocalizationNode:
    def __init__(self):
        rospy.init_node('aruco_localization_node', anonymous=True)

        self._vehicle_name = os.environ['VEHICLE_NAME']

        self.K = np.array([
            [332.6544121604296, 0.0, 316.81243765089346],
            [0.0, 329.6706637774356, 239.3524836292827],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.d = np.array([
            -0.3006822939569327,
            0.0674782489139842,
            -0.008971789541553121,
            0.0014690287922984812,
            0.0
        ], dtype=np.float32)

        self.marker_size = 0.05

        self.known_marker_poses = {
            33: (0.0, 1.1, 0.1, 0.0, 0.0, 0.0),
            26: (0.2, 1.1, 0.1, 0.0, 0.0, np.pi / 2),
        }

        self.bridge = CvBridge()

        # Relative localization için
        self.origin_world_pose = None              # ilk görülen absolute world pose
        self.last_pose = np.array([0.0, 0.0], dtype=np.float32)   # relative last pose
        self.current_pose = np.array([0.0, 0.0], dtype=np.float32)
        self.last_time = None
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.is_tag_visible = False

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.image_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/image/compressed",
            CompressedImage,
            self.image_callback,
            queue_size=1
        )

        self.pose_pub = rospy.Publisher(
            f"/{self._vehicle_name}/estimated_pose",
            PoseStamped,
            queue_size=10
        )

        self.cmd_pub = rospy.Publisher(
            f"/{self._vehicle_name}/cmd_vel",
            Twist,
            queue_size=10
        )

        self.cmd_timer = rospy.Timer(rospy.Duration(0.1), self.move_robot_callback)

        # cv2.namedWindow("ArUco Camera View", cv2.WINDOW_NORMAL)
        rospy.loginfo("ArUco Localization & Camera Visualization Node Initialized.")
        rospy.loginfo("Robot initial pose set to (0.0, 0.0). Waiting first tag for origin calibration.")

    def move_robot_callback(self, event):
        twist = Twist()
        twist.linear.x = 0.1
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def euler_to_matrix(self, roll, pitch, yaw, x, y, z):
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R = R_z @ R_y @ R_x

        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z
        return T

    def image_callback(self, msg):
        current_time = rospy.Time.now().to_sec()

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                rospy.logwarn("Decoded image is None.")
                return
        except Exception as e:
            rospy.logwarn(f"Image decode error: {e}")
            return

        display_image = cv_image.copy()
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        detected_pose = None
        dt = 0.0
        if self.last_time is not None:
            dt = current_time - self.last_time

        self.is_tag_visible = False

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(display_image, corners, ids)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.marker_size,
                self.K,
                self.d
            )

            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                cv2.drawFrameAxes(
                    display_image,
                    self.K,
                    self.d,
                    rvec,
                    tvec,
                    self.marker_size * 0.5
                )

                text = f"ID: {marker_id}  x:{tvec[0]:.2f} y:{tvec[1]:.2f} z:{tvec[2]:.2f}"
                corner_pt = tuple(corners[i][0][0].astype(int))
                cv2.putText(
                    display_image,
                    text,
                    (corner_pt[0], max(20, corner_pt[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                if marker_id in self.known_marker_poses:
                    T_world_marker = self.euler_to_matrix(*self.known_marker_poses[marker_id])

                    R_cam_marker, _ = cv2.Rodrigues(rvec)
                    T_cam_marker = np.eye(4)
                    T_cam_marker[:3, :3] = R_cam_marker
                    T_cam_marker[:3, 3] = tvec.reshape(3)

                    T_marker_cam = np.linalg.inv(T_cam_marker)
                    T_world_cam = T_world_marker @ T_marker_cam

                    absolute_world_x = T_world_cam[0, 3]
                    absolute_world_y = T_world_cam[1, 3]

                    # İlk tag görüldüğünde burayı başlangıç kabul et
                    if self.origin_world_pose is None:
                        self.origin_world_pose = np.array(
                            [absolute_world_x, absolute_world_y],
                            dtype=np.float32
                        )
                        rospy.loginfo(
                            f"Origin locked at absolute pose: "
                            f"({absolute_world_x:.2f}, {absolute_world_y:.2f}) -> relative (0.00, 0.00)"
                        )

                    # Relative pose hesapla
                    relative_x = absolute_world_x - self.origin_world_pose[0]
                    relative_y = absolute_world_y - self.origin_world_pose[1]

                    curr_pose = np.array([relative_x, relative_y], dtype=np.float32)

                    if dt > 0 and self.last_time is not None:
                        self.velocity = (curr_pose - self.last_pose) / dt

                    self.current_pose = curr_pose
                    self.last_pose = curr_pose
                    self.last_time = current_time
                    self.is_tag_visible = True
                    detected_pose = curr_pose

                    cv2.putText(
                        display_image,
                        f"REL X:{relative_x:.2f} Y:{relative_y:.2f}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )

                    cv2.putText(
                        display_image,
                        f"ABS X:{absolute_world_x:.2f} Y:{absolute_world_y:.2f}",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    break

        if detected_pose is not None:
            rospy.loginfo(f"RELATIVE POSE -> X: {detected_pose[0]:.2f}, Y: {detected_pose[1]:.2f}")

            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "world"
            pose_msg.pose.position.x = float(detected_pose[0])
            pose_msg.pose.position.y = float(detected_pose[1])
            pose_msg.pose.position.z = 0.0
            pose_msg.pose.orientation.w = 1.0

            self.pose_pub.publish(pose_msg)

        status_text = "TAG VISIBLE" if self.is_tag_visible else "NO TAG"
        cv2.putText(
            display_image,
            status_text,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if self.is_tag_visible else (0, 0, 255),
            2
        )

        if self.origin_world_pose is None:
            origin_text = "ORIGIN: waiting first tag"
        else:
            origin_text = f"ORIGIN ABS: ({self.origin_world_pose[0]:.2f}, {self.origin_world_pose[1]:.2f})"

        cv2.putText(
            display_image,
            origin_text,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # GUI disabled (No display on Duckiebot)
        # cv2.imshow("ArUco Camera View", display_image)
        # key = cv2.waitKey(1) & 0xFF

        # if key == ord('q'):
        #     rospy.signal_shutdown("User pressed q")

    def shutdown_hook(self):
        pass # cv2.destroyAllWindows()


def main():
    node = ArUcoLocalizationNode()
    rospy.on_shutdown(node.shutdown_hook)
    rospy.spin()


if __name__ == '__main__':
    main()

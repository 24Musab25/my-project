#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int8

class CamSubscriberNode:
    def __init__(self):
        rospy.init_node('cam_subscriber_node', anonymous=True)
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'duckiebot')

        # ArUco Config
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.mission_mode = 0  # 0: waiting, 1: ArUco, 2: Odometry Backup

        # Subscribers
        self.image_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/image/compressed",
            CompressedImage,
            self.image_callback,
            queue_size=1
        )
        self.mode_sub = rospy.Subscriber(
            "/duckie_mission/mission_mode",
            Int8,
            self.mode_callback,
            queue_size=1
        )
        
        # Publishers
        self.detect_pub = rospy.Publisher(
            "/duckie_mission/is_aruco_detected",
            Bool,
            queue_size=1
        )
        rospy.loginfo("Cam Subscriber Node Started")

    def mode_callback(self, msg):
        self.mission_mode = msg.data

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return
        except Exception as e:
            rospy.logwarn(f"Image decode error: {e}")
            return

        # Prepare display image and gray scale for detection
        display_image = cv_image.copy()
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        is_detected = (ids is not None and len(ids) > 0)
        
        # Publish Detection status
        self.detect_pub.publish(Bool(is_detected))

        # Visualization
        if is_detected:
            cv2.aruco.drawDetectedMarkers(display_image, corners, ids)
            cv2.putText(display_image, "ARUCO DETECTED", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_image, "NO ARUCO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        mode_str = "WAIT" if self.mission_mode == 0 else "ARUCO Pose" if self.mission_mode == 1 else "ODOM Pose"
        cv2.putText(display_image, f"MODE: {mode_str}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # You can uncomment these if you have GUI enabled. Otherwise it might crash Duckiebot Python.
        # cv2.imshow("Cam View", display_image)
        # cv2.waitKey(1)

def main():
    node = CamSubscriberNode()
    rospy.spin()

if __name__ == '__main__':
    main()

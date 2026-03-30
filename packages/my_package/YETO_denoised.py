#!/usr/bin/env python3

import os
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist


class ArUcoNode:
    def __init__(self):
        rospy.init_node("aruco_node")

        self.vehicle = os.environ['VEHICLE_NAME']

        # ---- Publishers ----
        self.cam_pub = rospy.Publisher(
            f"/{self.vehicle}/aruco/camera",
            CompressedImage,
            queue_size=1
        )

        self.map_pub = rospy.Publisher(
            f"/{self.vehicle}/aruco/map",
            CompressedImage,
            queue_size=1
        )

        self.pose_pub = rospy.Publisher(
            f"/{self.vehicle}/aruco/pose",
            PoseStamped,
            queue_size=10
        )

        self.cmd_pub = rospy.Publisher(
            f"/{self.vehicle}/car_cmd_switch_node/cmd",
            Twist,
            queue_size=1
        )

        # ---- Subscriber ----
        self.sub = rospy.Subscriber(
            f"/{self.vehicle}/camera_node/image/compressed",
            CompressedImage,
            self.callback,
            queue_size=1
        )

        # ---- ArUco ----
        self.dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
        self.params = cv2.aruco.DetectorParameters_create()

        # ---- State ----
        self.pose = np.array([0.0, 0.0])
        self.path = []

        self.map_size = 600
        self.scale = 200

        rospy.Timer(rospy.Duration(0.1), self.move)

        rospy.loginfo("Node started ✔")

    def move(self, _):
        msg = Twist()
        msg.linear.x = 0.2
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def draw_map(self):
        img = np.ones((self.map_size, self.map_size, 3), dtype=np.uint8) * 255
        center = self.map_size // 2

        for p in self.path:
            px = int(center + p[0] * self.scale)
            py = int(center - p[1] * self.scale)
            cv2.circle(img, (px, py), 2, (0, 0, 255), -1)

        return img

    def publish_image(self, img, pub):
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode(".jpg", img)[1]).tobytes()
        pub.publish(msg)

    def publish_pose(self):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = float(self.pose[0])
        msg.pose.position.y = float(self.pose[1])
        msg.pose.orientation.w = 1.0
        self.pose_pub.publish(msg)

    def callback(self, msg):
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except:
            return

        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.dict,
            parameters=self.params
        )

        display = frame.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

            # basit fake pose (örnek)
            self.pose += np.array([0.01, 0.0])
            self.path.append(self.pose.copy())

            # 📢 terminale yaz
            rospy.loginfo(f"X: {self.pose[0]:.2f}  Y: {self.pose[1]:.2f}")

            # 📡 pose publish
            self.publish_pose()

        # ---- map oluştur ----
        map_img = self.draw_map()

        # ---- publish ----
        self.publish_image(display, self.cam_pub)
        self.publish_image(map_img, self.map_pub)


if __name__ == "__main__":
    node = ArUcoNode()
    rospy.spin()

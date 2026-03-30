#!/usr/bin/env python3

import os
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped, Twist


class ArUcoLocalizationNode:
    def __init__(self):
        rospy.init_node("aruco_localization")

        self.vehicle = os.environ['VEHICLE_NAME']

        # ---- Publishers ----
        self.cam_pub = rospy.Publisher(f"/{self.vehicle}/aruco/camera", CompressedImage, queue_size=1)
        self.map_pub = rospy.Publisher(f"/{self.vehicle}/aruco/map", CompressedImage, queue_size=1)
        self.pose_pub = rospy.Publisher(f"/{self.vehicle}/aruco/pose", PoseStamped, queue_size=10)

        self.cmd_pub = rospy.Publisher(f"/{self.vehicle}/car_cmd_switch_node/cmd", Twist, queue_size=1)

        # ---- Subscriber ----
        rospy.Subscriber(
            f"/{self.vehicle}/camera_node/image/compressed",
            CompressedImage,
            self.callback,
            queue_size=1
        )

        # ---- Camera calibration ----
        self.K = np.array([
            [332.65, 0, 316.81],
            [0, 329.67, 239.35],
            [0, 0, 1]
        ], dtype=np.float32)

        self.d = np.array([-0.30, 0.06, 0, 0, 0], dtype=np.float32)

        # ---- Known markers ----
        self.marker_size = 0.05
        self.known_tags = {
            9: (0.0,0.3 , 0.0),
           # 26: (0.2, 1.1, np.pi/2)
        }

        # ---- State ----
        self.origin = None
        self.pose = np.array([0.0, 0.0])
        self.path = []

        self.map_size = 600
        self.scale = 200

        self.dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
        self.params = cv2.aruco.DetectorParameters_create()

        rospy.Timer(rospy.Duration(0.1), self.move)

        rospy.loginfo("ArUco localization started ✔")

    def move(self, _):
        cmd = Twist()
        cmd.linear.x = 0.2
        self.cmd_pub.publish(cmd)

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

    def draw_map(self):
        img = np.ones((self.map_size, self.map_size, 3), dtype=np.uint8) * 255
        c = self.map_size // 2

        for p in self.path:
            px = int(c + p[0]*self.scale)
            py = int(c - p[1]*self.scale)
            cv2.circle(img, (px, py), 2, (0, 0, 255), -1)

        return img

    def callback(self, msg):
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except:
            return

        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dict, parameters=self.params)

        display = frame.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.K, self.d
            )

            for i in range(len(ids)):
                tag_id = int(ids[i][0])

                if tag_id not in self.known_tags:
                    continue

                tvec = tvecs[i][0]

                # 🔥 Dünya koordinatı hesaplama (basitleştirilmiş)
                world_x, world_y, yaw = self.known_tags[tag_id]

                cam_x = tvec[0]
                cam_z = tvec[2]

                # robot pose (approx)
                robot_x = world_x - cam_z
                robot_y = world_y - cam_x

                if self.origin is None:
                    self.origin = np.array([robot_x, robot_y])

                self.pose = np.array([
                    robot_x - self.origin[0],
                    robot_y - self.origin[1]
                ])

                self.path.append(self.pose.copy())

                # 📢 log
                rospy.loginfo(f"POSE → X:{self.pose[0]:.2f} Y:{self.pose[1]:.2f}")

                self.publish_pose()

        map_img = self.draw_map()

        self.publish_image(display, self.cam_pub)
        self.publish_image(map_img, self.map_pub)


if __name__ == "__main__":
    ArUcoLocalizationNode()
    rospy.spin()

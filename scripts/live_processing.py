#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from insta360_ros_driver.tools import *

class LiveProcessing():
    def __init__(self):
        rospy.init_node('live_processing_node')
        self.bridge = CvBridge()

        # Parameters for topics, undistortion, and frame IDs
        self.topic_name = '/insta_image_yuv'
        self.undistort = rospy.get_param('undistort', False)
        self.K = np.asarray(rospy.get_param("K", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])).astype(np.float32)
        self.D = np.asarray(rospy.get_param("D", [0, 0, 0, 0])).astype(np.float32)

        # Define frame IDs for front and back cameras
        self.front_frame_id = 'front_camera_optical_frame'
        self.back_frame_id = 'back_camera_optical_frame'

        distortion_with_balance = False
        balance = 1.0
        image_scale = 1.0

        h, w = 1152, 2304  # Hardcoded image size for Insta360 X3
        if not distortion_with_balance:
            # Generate undistortion maps for front and back cameras with the same resolution
            self.map1_front, self.map2_front = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, (w//2, h), cv2.CV_32FC1)
            self.map1_back, self.map2_back = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, (w//2, h), cv2.CV_32FC1)
            width, height = w // 2, h
        else:
            # Original width and height
            original_width, original_height = w // 2, h
            # Increased resolution
            new_width, new_height = int(original_width * image_scale), int(original_height * image_scale)
            # Estimate new camera matrix for front and back cameras with increased resolution
            new_camera_matrix_front = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D, (original_width, original_height), np.eye(3), balance=balance, new_size=(new_width, new_height))
            new_camera_matrix_back = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D, (original_width, original_height), np.eye(3), balance=balance, new_size=(new_width, new_height))
            # Generate undistortion maps with the new 4x resolution
            self.map1_front, self.map2_front = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), new_camera_matrix_front, (new_width, new_height), cv2.CV_32FC1)
            self.map1_back, self.map2_back = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), new_camera_matrix_back, (new_width, new_height), cv2.CV_32FC1)
            width, height = new_width, new_height

        # Update the camera matrix and distortion coefficients to reflect the undistorted image
        self.K = new_camera_matrix_front
        self.D = np.zeros((5, 1))

        # Precompute camera info messages
        self.front_camera_info_msg = self.get_camera_info(width, height, self.K, self.D, self.front_frame_id)
        self.back_camera_info_msg = self.get_camera_info(width, height, self.K, self.D, self.back_frame_id)

        queue_size = 1
        self.latency = rospy.Duration.from_sec(0.2)  # 200 ms latency

        # Image subscribers and publishers
        self.image_sub = rospy.Subscriber(self.topic_name, Image, self.processing)
        self.front_image_pub = rospy.Publisher('front_camera/image_raw', Image, queue_size=queue_size)
        self.back_image_pub = rospy.Publisher('back_camera/image_raw', Image, queue_size=queue_size)

        # Camera info publishers
        self.front_camera_info_pub = rospy.Publisher('front_camera/camera_info', CameraInfo, queue_size=queue_size)
        self.back_camera_info_pub = rospy.Publisher('back_camera/camera_info', CameraInfo, queue_size=queue_size)

    def get_camera_info(self, width, height, K, D, frame_id):
        camera_info_msg = CameraInfo()

        # Camera resolution
        camera_info_msg.width = width
        camera_info_msg.height = height

        # Camera intrinsic matrix (K)
        camera_info_msg.K = K.flatten().tolist()

        # Distortion coefficients (D) - Explicitly cast to float
        camera_info_msg.D = [float(d) for d in D.flatten()]

        # Distortion model
        camera_info_msg.distortion_model = "plumb_bob"

        # Rectification matrix (identity for now)
        camera_info_msg.R = np.eye(3).flatten().tolist()

        # Projection matrix (P)
        # In this case, we can set it to [K, [0, 0, 0]]
        camera_info_msg.P = np.hstack((K, np.zeros((3, 1)))).flatten().tolist()

        # Set the frame ID for the camera
        camera_info_msg.header.frame_id = frame_id

        return camera_info_msg

    def processing(self, msg):
        try:
            current_timestamp = msg.header.stamp + self.latency

            # Convert ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Convert the YUV image to BGR format
            bgr_image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420).astype(np.uint8)

            # Assuming the image is horizontally split for Front | Back
            height, width = bgr_image.shape[:2]
            mid_point = width // 2

            front_image = bgr_image[:, :mid_point]
            back_image = bgr_image[:, mid_point:]
        
            # Live Undistortion
            if self.undistort:
                interpolation_method = cv2.INTER_LINEAR
                front_image = cv2.remap(front_image, self.map1_front, self.map2_front, interpolation=interpolation_method)
                back_image = cv2.remap(back_image, self.map1_back, self.map2_back, interpolation=interpolation_method)

            # Convert to raw Image message and set the frame ID
            front_image_msg = self.bridge.cv2_to_imgmsg(front_image, encoding="bgr8")
            back_image_msg = self.bridge.cv2_to_imgmsg(back_image, encoding="bgr8")
            front_image_msg.header.frame_id = self.front_frame_id
            back_image_msg.header.frame_id = self.back_frame_id

            # Publish the CameraInfo messages for front and back cameras
            front_camera_info_msg = self.front_camera_info_msg
            back_camera_info_msg = self.back_camera_info_msg

            # Set the timestamp for the images and camera info
            front_image_msg.header.stamp = current_timestamp
            back_image_msg.header.stamp = current_timestamp
            front_camera_info_msg.header.stamp = current_timestamp
            back_camera_info_msg.header.stamp = current_timestamp

            # Publish the images and camera info
            self.front_image_pub.publish(front_image_msg)
            self.front_camera_info_pub.publish(front_camera_info_msg)
            self.back_image_pub.publish(back_image_msg)
            self.back_camera_info_pub.publish(back_camera_info_msg)

        except CvBridgeError as e:
            rospy.logerr("CvBridgeError: %s", e)
        except Exception as e:
            rospy.logerr("Failed to process image: %s", e)

if __name__ == '__main__':
    try:
        live_processing = LiveProcessing()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
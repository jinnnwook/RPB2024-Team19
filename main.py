# !/usr/bin/env python3
import rclpy
import numpy as np
import cv2
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header

def detect(image):
    img=image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask_black = cv2.inRange(hsv_image, lower_black, upper_black)
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        mask_inner = np.zeros_like(mask_black, dtype=np.uint8)
        cv2.drawContours(mask_inner, [cnt], -1, color=255, thickness=-1)
        mask_inner = mask_inner > 0
        mask_red = ((hsv_image[:, :, 0] > 170) | (hsv_image[:, :, 0] < 10)) & (hsv_image[:, :, 1] > 50) & (hsv_image[:, :, 2] > 50)
        mask_green = (hsv_image[:, :, 0] > 50) & (hsv_image[:, :, 0] < 70) & (hsv_image[:, :, 1] > 50) & (hsv_image[:, :, 2] > 50)
        mask_blue = (hsv_image[:, :, 0] > 110) & (hsv_image[:, :, 0] < 130) & (hsv_image[:, :, 1] > 50) & (hsv_image[:, :, 2] > 50)
        red_pixels = np.sum(mask_red & mask_inner)
        green_pixels = np.sum(mask_green & mask_inner)
        blue_pixels = np.sum(mask_blue & mask_inner)
        if red_pixels > green_pixels and red_pixels > blue_pixels:
            return "R"
        elif green_pixels > blue_pixels:
            return "G"
        else:
            return "B"

class DetermineColor(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.callback, 10)
        self.color_pub = self.create_publisher(Header, '/rotate_cmd', 10)
        self.bridge = CvBridge()
        #self.count = 0

    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            msg = Header()
            msg = data.header
            msg.frame_id = '0'  # default: STOP
            c=detect(image)
            if c=='B':
                msg.frame_id='+1'
            elif c=='R':
                msg.frame_id='-1'
            else:
                msg.frame_id='0'
            self.color_pub.publish(msg)
        except CvBridgeError as e:
            self.get_logger().error('Failed to convert image: %s' % e)


if __name__ == '__main__':
    rclpy.init()
    detector = DetermineColor()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()
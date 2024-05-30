# !/usr/bin/env python3
import rclpy
import numpy as np
import cv2
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header

def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 7)
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_count=0
    blue_count=0
    green_count=0
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if rect[1][0] > 50 and rect[1][1] > 50:
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
            width, height = int(rect[1][0]), int(rect[1][1])
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1],[0, 0],[width-1, 0],[width-1, height-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (width, height))
            hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 51, 51])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 51, 51])
            upper_red2 = np.array([180, 255, 255])
            lower_green = np.array([35, 51, 51])
            upper_green = np.array([75, 255, 255])
            lower_blue = np.array([95, 51, 51])
            upper_blue = np.array([120, 255, 255])
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            red_count += cv2.countNonZero(red_mask)
            green_count += cv2.countNonZero(green_mask)
            blue_count += cv2.countNonZero(blue_mask)
    if red_count > green_count and red_count > blue_count:
        return 'R'
    elif green_count > red_count and green_count > blue_count:
        return 'G'
    else:
        return 'B'

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

import cv2
import random
import numpy as np
from typing import Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import message_filters
from cv_bridge import CvBridge
from ultralytics.utils.plotting import Annotator, colors

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from human_msgs.msg import KeyPoint2D
from human_msgs.msg import KeyPoint3D
from human_msgs.msg import Detection
from human_msgs.msg import DetectionArray

class DebugNode(Node):

    def __init__(self) -> None:
        super().__init__("debug_node")
        
        #Parameters
        self._class_to_color = {}
        image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        #CV Bridge
        self.cv_bridge = CvBridge()
        
        #Publisher
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)
        self._kp_markers_pub = self.create_publisher(
            MarkerArray, "dgb_kp_markers", 10)
        
        #Subscriber
        image_sub = message_filters.Subscriber(
            self, Image, "/camera/image_raw", qos_profile=image_qos_profile)
        detections_sub = message_filters.Subscriber(
            self, DetectionArray, "/detections", qos_profile=10)
        detections_3d_sub = message_filters.Subscriber(
            self, DetectionArray, "/detections_3d", qos_profile=10)
        

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (image_sub, detections_sub, detections_3d_sub), 10, 0.5)
        self._synchronizer.registerCallback(self.detections_cb)
        
    def detections_cb(self, img_msg: Image, detection_msg: DetectionArray, detections3d_msg: DetectionArray) -> None:

        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)
        kp_marker_array = MarkerArray()

        detection: Detection
        for detection in detection_msg.detections:

            cv_image = self.draw_keypoints(cv_image, detection)
            
        detection_3d: Detection
        for detection_3d in detections3d_msg.detections:
            if detection_3d.keypoints3d.frame_id:
                for kp in detection_3d.keypoints3d.data:
                    marker = self.create_kp_marker(kp)
                    marker.header.frame_id = detection_3d.keypoints3d.frame_id
                    marker.header.stamp = img_msg.header.stamp
                    marker.id = len(kp_marker_array.markers)
                    kp_marker_array.markers.append(marker)

        # publish dbg image
        self._dbg_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image,
                                                           encoding=img_msg.encoding))
        self._kp_markers_pub.publish(kp_marker_array)
        
    def draw_keypoints(self, cv_image: np.array, detection: Detection) -> np.array:

        keypoints_msg = detection.keypoints

        ann = Annotator(cv_image)

        kp: KeyPoint2D
        for kp in keypoints_msg.data:
            color_k = [int(x) for x in ann.kpt_color[kp.id - 1]
                       ] if len(keypoints_msg.data) == 17 else colors(kp.id - 1)

            cv2.circle(cv_image, (int(kp.point.x), int(kp.point.y)),
                       5, color_k, -1, lineType=cv2.LINE_AA)

        def get_pk_pose(kp_id: int) -> Tuple[int]:
            for kp in keypoints_msg.data:
                if kp.id == kp_id:
                    return (int(kp.point.x), int(kp.point.y))
            return None

        for i, sk in enumerate(ann.skeleton):
            kp1_pos = get_pk_pose(sk[0])
            kp2_pos = get_pk_pose(sk[1])

            if kp1_pos is not None and kp2_pos is not None:
                cv2.line(cv_image, kp1_pos, kp2_pos, [
                    int(x) for x in ann.limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

        return cv_image
    
    def create_kp_marker(self, keypoint: KeyPoint3D) -> Marker:

        marker = Marker()

        marker.ns = "yolov8_3d"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.frame_locked = False

        marker.pose.position.x = keypoint.point.x
        marker.pose.position.y = keypoint.point.y
        marker.pose.position.z = keypoint.point.z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.b = keypoint.score * 255.0
        marker.color.g = 0.0
        marker.color.r = (1.0 - keypoint.score) * 255.0
        marker.color.a = 0.4

        marker.lifetime = Duration(seconds=0.5).to_msg()
        marker.text = f"ID: {keypoint.id}, Score: {keypoint.score:.2f}"

        return marker

def main():
    rclpy.init()
    node = DebugNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
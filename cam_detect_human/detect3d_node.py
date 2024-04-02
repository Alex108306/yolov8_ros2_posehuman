import numpy as np
from typing import List, Tuple

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import message_filters
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener
import pyrealsense2 as rs

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped, Point
from human_msgs.msg import Detection
from human_msgs.msg import DetectionArray
from human_msgs.msg import KeyPoint3D
from human_msgs.msg import KeyPoint3DArray
from human_msgs.msg import BoundingBox3D
from people_msgs.msg import People, Person
from leg_detector_msgs.msg import HumanArray, Human

class Detect3d(Node):
    
    def __init__(self) -> None:
        super().__init__("keypoint_3d")
        
        #Parameter
        self.target_frame = "map"
        self.maximum_detection_thres = 0.7
        self.depth_image_units_divisor = 1000 # To transform into meter from mili meter
        depth_image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        depth_info_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        #Transform and Bridge
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cv_bridge = CvBridge()
        
        #Publish
        self._pub = self.create_publisher(DetectionArray, "/detections_3d", 10)
        self.human_pub = self.create_publisher(HumanArray, "/people_tracked_cam", 10)
        
        #Subscribtion
        self.depth_sub = message_filters.Subscriber(
            self, Image, "/camera/depth/image_raw",
            qos_profile=depth_image_qos_profile)
        self.depth_info_sub = message_filters.Subscriber(
            self, CameraInfo, "/camera/depth/camera_info",
            qos_profile=depth_info_qos_profile)
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "/tracking")

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.depth_sub, self.depth_info_sub, self.detections_sub), 10, 0.5)
        self._synchronizer.registerCallback(self.on_detections)
        
    def on_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        detections_msg: DetectionArray,
    ) -> None:
        new_detections_msg = DetectionArray()
        new_detections_msg.header = detections_msg.header
        new_detections_msg.detections = self.process_detections(
            depth_msg, depth_info_msg, detections_msg)
        self.CalculatePose(new_detections_msg)
        self._pub.publish(new_detections_msg)
        
    def process_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        detections_msg: DetectionArray
    ) -> List[Detection]:
        
        if not detections_msg.detections:
            return []

        transform = self.get_transform(depth_info_msg.header.frame_id)

        if transform is None:
            return []

        new_detections = []
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg)
        
        for detection in detections_msg.detections:
            bbox3d = self.convert_bb_to_3d(
                depth_image, depth_info_msg, detection)

            if bbox3d is not None:
                new_detections.append(detection)

                bbox3d = Detect3d.transform_3d_box(
                    bbox3d, transform[0], transform[1])
                bbox3d.frame_id = self.target_frame
                new_detections[-1].bbox3d = bbox3d
                # if detection.keypoints.data:
                #     new_detections.append(detection)
                #     keypoints3d = self.convert_keypoints_to_3d(
                #         depth_image, depth_info_msg, detection)
                #     keypoints3d = Detect3d.transform_3d_keypoints(
                #         keypoints3d, transform[0], transform[1])
                #     keypoints3d.frame_id = self.target_frame
                #     new_detections[-1].keypoints3d = keypoints3d    
        
        return new_detections
    
    def convert_bb_to_3d(
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection
    ) -> BoundingBox3D:

        # crop depth image by the 2d BB
        center_x = int(detection.bbox.center.position.x)
        center_y = int(detection.bbox.center.position.y)
        size_x = int(detection.bbox.size.x)
        size_y = int(detection.bbox.size.y)

        bb_center_z_coord = depth_image.item(center_y, center_x)
        print(f"z: {bb_center_z_coord}")

        # project from image to world space
        k = depth_info.k
        px, py, fx, fy = k[2], k[5], k[0], k[4]
        x = bb_center_z_coord * (center_x - px) / fx
        y = bb_center_z_coord * (center_y - py) / fy
        w = bb_center_z_coord * (size_x / fx)
        h = bb_center_z_coord * (size_y / fy)

        # create 3D BB
        msg = BoundingBox3D()
        msg.center.position.x = x
        msg.center.position.y = y
        msg.center.position.z = bb_center_z_coord
        msg.size.x = w
        msg.size.y = h

        return msg
    
    def convert_keypoints_to_3d(
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection
    ) -> KeyPoint3DArray:
        
        height, width = depth_image.shape
        msg_array = KeyPoint3DArray()
        
        if detection.keypoints.data:
            for p in detection.keypoints.data:
                msg = KeyPoint3D()
                u = int(round(p.point.x))
                v = int(round(p.point.y))
                u = max(0, min(u, width - 1))
                v = max(0, min(v, height - 1))
                z = depth_image.item(v,u)
                k = depth_info.k
                px, py, fx, fy = k[2], k[5], k[0], k[4]
                x = z * (p.point.x - px) / fx
                y = z * (p.point.y - py) / fy
                msg.point.x = x
                msg.point.y = y
                msg.point.z = float(z)
                msg.id = p.id
                msg.score = p.score
                msg_array.data.append(msg)
        
        return msg_array
    
    def get_transform(self, frame_id: str) -> Tuple[np.ndarray]:
        # transform position from image frame to target_frame
        rotation = None
        translation = None

        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame,
                frame_id,
                rclpy.time.Time())

            translation = np.array([transform.transform.translation.x,
                                    transform.transform.translation.y,
                                    transform.transform.translation.z])

            rotation = np.array([transform.transform.rotation.w,
                                 transform.transform.rotation.x,
                                 transform.transform.rotation.y,
                                 transform.transform.rotation.z])

            return translation, rotation

        except TransformException as ex:
            self.get_logger().error(f"Could not transform: {ex}")
            return None
        
    def CalculatePose(self, detectionarray: DetectionArray):
        human_array = HumanArray()
        human_array.number = len(detectionarray.detections)
        print(f"number: {len(detectionarray.detections)}")
        for detection in detectionarray.detections:
            # human_id = 0
            # if detection.keypoints3d.data:
            #     point_left_shoulder = Point()
            #     point_right_shoulder = Point()
            #     pose_human = Point()
            #     for i in range(len(detection.keypoints3d.data)):
            #         if detection.keypoints3d.data[i].id == 1:
            #             continue
            #         if detection.keypoints3d.data[i].id == 6:
            #             point_left_shoulder.x=detection.keypoints3d.data[i].point.x
            #             point_left_shoulder.y=detection.keypoints3d.data[i].point.y
            #             point_left_shoulder.z=detection.keypoints3d.data[i].point.z
            #         if detection.keypoints3d.data[i].id == 7:
            #             point_right_shoulder.x=detection.keypoints3d.data[i].point.x
            #             point_right_shoulder.y=detection.keypoints3d.data[i].point.y
            #             point_right_shoulder.z=detection.keypoints3d.data[i].point.z
            #         print(f"y1: {point_left_shoulder.y}, y2: {point_right_shoulder.y}")
            #     pose_human.x = (point_left_shoulder.x + point_right_shoulder.x) * 1.034 / 2
            #     pose_human.y = (point_left_shoulder.y + point_right_shoulder.y) * 1.023 / 2
            #     pose_human.z = (point_left_shoulder.z + point_right_shoulder.z) * 1.03 / 2
            pose_human = Point()
            pose_human.x = detection.bbox3d.center.position.x * 1.034
            pose_human.y = detection.bbox3d.center.position.y * 1.023
            pose_human.z = detection.bbox3d.center.position.z * 1.03
        
            
            #Publish the pose 
            human = Human()
            # human.id = detection.id
            human.pose.position.x = pose_human.x 
            human.pose.position.y = pose_human.y 
            human.pose.position.z = pose_human.z 
            human_array.people.append(human)
            
            print(f"pose_human : {pose_human}")
                
                # A_x = point_right_shoulder.x - point_left_shoulder.x
                # A_y = point_right_shoulder.y - point_left_shoulder.y
                
                # # Reference vector B pointing upwards
                # B_x = 0
                # B_y = 1
                
                # # Calculate the angle using atan2
                # # Note: atan2 receives "y" first, then "x" to compute the angle from the positive X-axis counterclockwise
                # theta_radians = math.atan2(A_y, A_x) - math.atan2(B_y, B_x)
                
                # # Normalize the angle to be between 0 and 2*pi
                # theta_radians = theta_radians % (2 * math.pi)
                
                # # Convert to degrees
                # theta_degrees = math.degrees(theta_radians)
                
                # # Ensure the angle is between 0 and 360 degrees
                # theta_degrees = theta_degrees % 360
                
                # print(f"Orientation of the person: {theta_degrees} degrees")
        
        self.human_pub.publish(human_array)
                
    @staticmethod
    def transform_3d_box(
        bbox: BoundingBox3D,
        translation: np.ndarray,
        rotation: np.ndarray
    ) -> BoundingBox3D:

        # position
        position = Detect3d.qv_mult(
            rotation,
            np.array([bbox.center.position.x,
                      bbox.center.position.y,
                      bbox.center.position.z])
        ) + translation

        bbox.center.position.x = position[0]
        bbox.center.position.y = position[1]
        bbox.center.position.z = position[2]

        # size
        size = Detect3d.qv_mult(
            rotation,
            np.array([bbox.size.x,
                      bbox.size.y,
                      bbox.size.z])
        )

        bbox.size.x = abs(size[0])
        bbox.size.y = abs(size[1])
        bbox.size.z = abs(size[2])

        return bbox               
        
    @staticmethod
    def transform_3d_keypoints(
        keypoints: KeyPoint3DArray,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> KeyPoint3DArray:

        for point in keypoints.data:
            position = Detect3d.qv_mult(
                rotation,
                np.array([
                    point.point.x,
                    point.point.y,
                    point.point.z
                ])
            ) + translation

            point.point.x = position[0]
            point.point.y = position[1]
            point.point.z = position[2]

        return keypoints

    @staticmethod
    def qv_mult(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = np.array(q, dtype=np.float64)
        v = np.array(v, dtype=np.float64)
        qvec = q[1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2 * (uv * q[0] + uuv)
    
def main():
    rclpy.init()
    node = Detect3d()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
        
        
import os
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.engine.results import Results
from ultralytics.engine.results import Keypoints
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from human_msgs.msg import Point2D
from human_msgs.msg import BoundingBox2D
from human_msgs.msg import KeyPoint2D
from human_msgs.msg import KeyPoint2DArray
from human_msgs.msg import Detection
from human_msgs.msg import DetectionArray
from typing import List, Tuple, Dict
import numpy as np


import cv2
    
class EstimateHuman(Node):
    
    def __init__(self) ->None:
        super().__init__("yolov8_node")
        
        #Parameters
        
        image_qos_profile = QoSProfile(
            reliability= QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        #CV Bridge
        self.cv_bridge = CvBridge()
        
        #YOLO
        package_name = 'cam_detect_human'
        package_directory = get_package_share_directory(package_name)
        model_name = 'yolov8m-pose.pt'
        model = os.path.join(package_directory, 'models', model_name)
        self.device = 'cuda:0'
        self.threshold = 0.5
        self.yolo = YOLO(model)
        self.yolo.fuse()
        
        #Publisher
        self.img_pub = self.create_publisher(Image, '/yolo/image', 10)
        self._pub = self.create_publisher(DetectionArray, "/detections", 10)
        
        #Subscriber
        self.img_raw = self.create_subscription(Image, '/camera/image_raw', self.image_callback, image_qos_profile)

    def image_callback(self, msg: Image):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
        self.height, self.width, _ = cv_image.shape
        results = self.yolo.predict(
            source=cv_image,
            verbose=False,
            stream=False,
            device=self.device,
            conf=self.threshold
        )
        # print(f'len:{len(results)}')
        # print(f"{len(results[0])}")
        if len(results[0]) != 0:
            results: Results
            results = results[0].cpu()
            
        else :
            image = self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.img_pub.publish(image) #Debug
            detection_msg = DetectionArray()
            detection_msg.header = msg.header
            detection_msg.number = len(results[0])
            print(f"number: {detection_msg.number}")
            self._pub.publish(detection_msg)
            return
        
        # print(f"width:{msg.width}, height:{msg.height}")
        
        #Get keypoints
        keypoints = self.get_keypoints(results)
        frame = results.plot()
        frame_msg = self.cv_bridge.cv2_to_imgmsg(frame, "bgr8")
        
        #Get boxes
        hypothesis = self.parse_hypothesis(results)
        boxes = self.parse_boxes(results)
        
        #Create detections message
        detection_msg = DetectionArray()
        detection_msg.number = 0
        
        for i in range(len(results)):
            
            if(boxes[i].center.position.x <= 580 and boxes[i].center.position.x >= 60 and boxes[i].center.position.y <= 420 and boxes[i].center.position.y >= 60):
                aux_msg = Detection()
                
                print(f"x: {boxes[i].center.position.x}, y:{boxes[i].center.position.y}")
                detection_msg.number = len(results)

                aux_msg.class_id = hypothesis[i]["class_id"]
                aux_msg.class_name = hypothesis[i]["class_name"]
                aux_msg.score = hypothesis[i]["score"]
                aux_msg.bbox = boxes[i]

                aux_msg.keypoints = keypoints[i]

                detection_msg.detections.append(aux_msg)
        
        print(f"number: {detection_msg.number}")
        detection_msg.header = msg.header
        # self.show_keypoints(msg, detection_msg)
        self._pub.publish(detection_msg)
        # self.img_pub.publish(frame_msg) #debug
        
    def get_keypoints(self, results: Results) -> List[KeyPoint2DArray]:
        keypoint_array_list = []
        points : Keypoints
        
        for points in results.keypoints:
            keypoint_array = KeyPoint2DArray()
            if points.conf is None:
                continue
            
            #Choose specific keypoints
            keypoint_indices = [0, 5, 6] # 0 nose, 3 left ear, 5 left shoulder, 4 right ear, 6 right shoulder
            
            for kp_id in keypoint_indices:
                p = points.xy[0][kp_id]
                conf = points.conf[0][kp_id]
                if conf >= self.threshold:
                    point_ = KeyPoint2D()
                    point_.id = kp_id + 1
                    point_.point.x = float(p[0])
                    point_.point.y = float(p[1])
                    point_.score = float(conf)
                    # print(f"x_kp: {point_.point.x}, y_kp: {point_.point.y}")
                
                    keypoint_array.data.append(point_)
        
            keypoint_array_list.append(keypoint_array)
        
        return keypoint_array_list
    
    def parse_hypothesis(self, results: Results) -> List[Dict]: # Dict like searching dictionary gg, ex: call list[""]

        hypothesis_list = []

        box_data: Boxes
        for box_data in results.boxes:
            hypothesis = {
                "class_id": int(box_data.cls),
                "class_name": self.yolo.names[int(box_data.cls)],
                "score": float(box_data.conf)
            }
            hypothesis_list.append(hypothesis)

        return hypothesis_list
    
    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:

        boxes_list = []

        box_data: Boxes
        for box_data in results.boxes:

            msg = BoundingBox2D()

            # get boxes values
            box = box_data.xywh[0]
            msg.center.position.x = float(box[0])
            msg.center.position.y = float(box[1])
            # print(f"x_box: {msg.center.position.x}, y_box: {msg.center.position.y} ")
            msg.size.x = float(box[2])
            msg.size.y = float(box[3])

            # append msg
            boxes_list.append(msg)

        return boxes_list
    
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
    
    def show_keypoints(self, img_msg: Image, detection_msg: DetectionArray) -> None:

        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)

        detection: Detection
        for detection in detection_msg.detections:

            cv_image = self.draw_keypoints(cv_image, detection)
            
        self.img_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image,
                                                           encoding=img_msg.encoding))

def main():
    rclpy.init()
    node = EstimateHuman()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

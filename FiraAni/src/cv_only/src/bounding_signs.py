#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import DBSCAN
import torch
import torchvision.transforms as transforms
import timm

class ImageProcessor:
    def __init__(self, camera_topic, model_path):
        self.bridge = CvBridge()
        self.camera_topic = camera_topic
        self.image_show_topic = "/image_show"
        self.model = self.load_model(model_path)
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        self.image_pub = rospy.Publisher(self.image_show_topic, Image, queue_size=2)
        self.processed_images = None

    def load_model(self, model_path):
        model = timm.create_model("xception", pretrained=True)
        in_features = model.get_classifier().in_features
        num_classes = 6
        model.fc = torch.nn.Linear(in_features, num_classes)
        checkpoints = torch.load(model_path)
        model.load_state_dict(checkpoints)
        model.eval()
        return model

    def image_callback(self, ros_image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        self.processed_images = cv_image  
        cv2.imwrite("/home/umic/Downloads/acb.png", self.processed_images)
        processed_images = self.process_first_code()

        if processed_images is not None:  
            if len(processed_images) == 1 and processed_images[0][1] == "open_regions_image":
                final_image = self.process_second_code(processed_images[0][0])
                self.send_to_ml_model(final_image)
            else:
                for img, name in processed_images:
                    self.send_to_ml_model(img)

    def process_first_code(self):
        processed_images = self.processed_images
        hsv_image = cv2.cvtColor(processed_images, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([0, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 55, 255])

        mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
        mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

        kernel = np.ones((5, 5), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

        blue_red_image, blue_red_result = self.process_and_save(mask_blue, mask_red, "blue", "red", processed_images)
        blue_white_image, blue_white_result = self.process_and_save(mask_blue, mask_white, "blue", "white", processed_images)
        red_white_image, red_white_result = self.process_and_save(mask_red, mask_white, "red", "white", processed_images)

        results = [] 

        if blue_white_result is not None:
            results.append((blue_white_result, "open_regions_image"))
        if blue_red_result is not None:
            results.append((blue_red_result, "blue_red_image"))
        if red_white_result is not None:
            results.append((red_white_result, "red_white_image"))

        for img, _ in results:
            img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            self.image_pub.publish(img_msg)

        return results

    def process_and_save(self, mask_primary, mask_check, color_name_primary, color_name_check, processed_images):
        masked_image_primary = cv2.bitwise_and(processed_images, processed_images, mask=mask_primary)
        gray_masked_image_primary = cv2.cvtColor(masked_image_primary, cv2.COLOR_BGR2GRAY)
        non_zero_points_primary = np.column_stack(np.where(gray_masked_image_primary > 0))

        if non_zero_points_primary.shape[0] == 0:
            print(f"No valid points found for {color_name_primary}.")
            return None, None

        clustering = DBSCAN(eps=10, min_samples=50).fit(non_zero_points_primary)
        labels = clustering.labels_

        mask_result = np.zeros(mask_primary.shape, dtype=np.uint8)
        unique_labels = set(labels)
        filtered_labels = [label for label in unique_labels if label != -1 and np.sum(labels == label) > 100]

        for label in filtered_labels:
            label_mask = (labels == label)
            cluster_points = non_zero_points_primary[label_mask]

            x_min, y_min = np.min(cluster_points, axis=0)
            x_max, y_max = np.max(cluster_points, axis=0)

            region_check_mask = mask_check[x_min:x_max, y_min:y_max]
            if np.any(region_check_mask > 0):
                mask_result[x_min:x_max, y_min:y_max] = 1

        binary_mask = mask_result.astype(np.uint8)
        final_result = cv2.bitwise_and(processed_images, processed_images, mask=binary_mask)

        if self.is_sign_within_bounds(final_result, (x_min, y_min, x_max, y_max)):
            return binary_mask, final_result
        else:
            return None, None

    def is_sign_within_bounds(self, image, bbox):
        x_min, y_min, x_max, y_max = bbox
        expected_height_range = (8, 16)  # cm
        expected_distance_range = (45, 55)  # cm

        camera_height = 21  # cm
        image_height, image_width = image.shape[:2]

        height_ratio = (camera_height - expected_height_range[1]) / camera_height
        width_ratio = expected_distance_range[1] / (image_width * 0.1) 

        height_bounds = (int(image_height * height_ratio), image_height)
        width_bounds = (0, int(image_width * width_ratio))

        if (height_bounds[0] <= y_min <= height_bounds[1] and
            width_bounds[0] <= x_min <= width_bounds[1] and
            height_bounds[0] <= y_max <= height_bounds[1] and
            width_bounds[0] <= x_max <= width_bounds[1]):
            return True
        else:
            return False

    def process_second_code(self, blue_white_image):
        hsv_image = cv2.cvtColor(blue_white_image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([0, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
        mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        kernel = np.ones((5, 5), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

        masked_image_blue = cv2.bitwise_and(blue_white_image, blue_white_image, mask=mask_blue)
        gray_masked_image_blue = cv2.cvtColor(masked_image_blue, cv2.COLOR_BGR2GRAY)
        non_zero_points_blue = np.column_stack(np.where(gray_masked_image_blue > 0))

        clustering = DBSCAN(eps=10, min_samples=50).fit(non_zero_points_blue)
        labels = clustering.labels_

        mask_result = np.zeros(mask_blue.shape, dtype=np.uint8)
        unique_labels = set(labels)
        filtered_labels = [label for label in unique_labels if label != -1 and np.sum(labels == label) > 100]

        for label in filtered_labels:
            label_mask = (labels == label)
            cluster_points = non_zero_points_blue[label_mask]

            x_min, y_min = np.min(cluster_points, axis=0)
            x_max, y_max = np.max(cluster_points, axis=0)

            region_red_mask = mask_red[x_min:x_max, y_min:y_max]
            if np.any(region_red_mask > 0):
                mask_result[x_min:x_max, y_min:y_max] = 0
            else:
                mask_result[x_min:x_max, y_min:y_max] = 1

        binary_mask = mask_result.astype(np.uint8)
        final_result = cv2.bitwise_and(blue_white_image, blue_white_image, mask=binary_mask)

        img_msg = self.bridge.cv2_to_imgmsg(final_result, "bgr8")
        self.image_pub.publish(img_msg)

        return final_result

    def send_to_ml_model(self, image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)

        class_names = ["dead_end_y", "no_entry_y", "proceed_forward_y", "proceed_left_y", "proceed_right_y", "stop_y"]
        print(predicted.item())
        class_name = class_names[predicted.item()]

        print("ML model output:", class_name)

if __name__ == "__main__":
    rospy.init_node('image_processor_node', anonymous=True)
    processor = ImageProcessor(camera_topic='/camera/color/image_rect_color', model_path="/home/umic/ws/src/cv_only/src/xception_trained_model.pth")
    rospy.spin()

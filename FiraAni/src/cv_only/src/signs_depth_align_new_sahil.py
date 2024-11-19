#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String , Float64 , Bool
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import DBSCAN
import torch
import torchvision.transforms as transforms
import timm
import time 


class ImageProcessor:
    def __init__(self, camera_topic, depth_topic, model_path):
        self.bridge = CvBridge()
        self.camera_topic = camera_topic
        self.depth_topic = depth_topic
        self.ml_topic = "/ml_image"
        self.model = self.load_model(model_path)
        self.ml_processed = rospy.Publisher(self.ml_topic, String, queue_size=1)
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
        self.image_processed_pub = rospy.Publisher("/crop_image", Image, queue_size=1)
        self.processed_images = None
        self.depth_image = None 
        self.sign_array = []
        self.ult_list = []
        self.sign = None 
        self.flag = 1
        self.sleep_flag = None 

        self.turn_radius_pub = rospy.Publisher("/turn_rad" , Float64 , queue_size=10)
        self.nothing_pub = rospy.Publisher("/nothing" , Bool , queue_size=10)

        self.gamma = rospy.get_param("/GAMMA")
        self.low_bound = rospy.get_param("/LOW_BOUND")
        self.upper_bound = rospy.get_param("/UP_BOUND")
        print("GAMMA: ", self.gamma)

    def mod_signs(self, array):
        
        frequency_dict = {}
        max_frequency = 0
        most_frequent = []

        for string in array:
            if string in frequency_dict:
                frequency_dict[string] += 1
            else:
                frequency_dict[string] = 1
        
            if frequency_dict[string] > max_frequency:
                max_frequency = frequency_dict[string]
                most_frequent = [string]
            elif frequency_dict[string] == max_frequency:
                most_frequent.append(string)
        
        return most_frequent
        
    def load_model(self, model_path):
        #rospy.loginfo("start hoogaya load model")
        model = timm.create_model("xception", pretrained=True)
        in_features = model.get_classifier().in_features
        num_classes = 6
        model.fc = torch.nn.Linear(in_features, num_classes)
        checkpoints = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoints)
        model.cuda()
        model.eval()
        #rospy.loginfo("end hoogaya load model")
        return model

    def depth_callback(self, depth_image):
        #rospy.loginfo("depth call_back start hoogaya")
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_image)
            self.depth_image = np.roll(self.depth_image, -0, axis=1)
            #rospy.loginfo("Depth image received, shifted, and processed.")
        except CvBridgeError as e:
            rospy.logerr("CvBridgeError in depth_callback: {}".format(e))
            
    def image_callback(self, ros_image):
        # rospy.loginfo("img call_back start hoogaya")
        if self.sleep_flag == 1 :
            rospy.loginfo(self.sleep_flag)
            rospy.sleep(15)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            inv_gamma = 1.0 / self.gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            cv_image = cv2.LUT(cv_image, table)
            #rospy.loginfo("RGB image received and processed.")
        except CvBridgeError as e:
            rospy.logerr("CvBridgeError in image_callback: {}".format(e))
            return

        self.processed_images = cv_image
        if self.depth_image is not None:
            #rospy.loginfo("Processing images...")
            processed_images = self.process_first_code()
            if processed_images is not None:
                for img, name in processed_images:
                    if name == "open_regions_image":
                        final_image = self.process_second_code(img)
                        self.process_and_send_to_ml_model(final_image)
                    else:
                        self.process_and_send_to_ml_model(img)

    def process_first_code(self):
        if self.sleep_flag == 1 : 
            rospy.sleep(15)
        
        rospy.loginfo("processing the first code")
        hsv_image = cv2.cvtColor(self.processed_images, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([0, 255, 255])
        lower_red2 = np.array([170, 150, 50])
        upper_red2 = np.array([180, 255, 255])
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 35, 255])

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

        depth_mask = (self.depth_image > self.low_bound) & (self.depth_image < self.upper_bound)
        depth_mask = depth_mask.astype(np.uint8) * 255
        mask_blue = cv2.bitwise_and(mask_blue, mask_blue, mask=depth_mask)
        mask_red = cv2.bitwise_and(mask_red, mask_red, mask=depth_mask)
        mask_white = cv2.bitwise_and(mask_white, mask_white, mask=depth_mask)
        
        #cv2.imshow("edf", mask_red)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
	#print(" first_code")

        blue_red_image, blue_red_result = self.process_and_save(mask_blue, mask_red, "blue", "red", self.processed_images)
        blue_white_image, blue_white_result = self.process_and_save(mask_blue, mask_white, "blue", "white", self.processed_images)
        red_white_image, red_white_result = self.process_and_save(mask_red, mask_white, "red", "white", self.processed_images)

        #cv2.imshow("edf", red_white_result)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        results = []
        
        if blue_white_result is not None:
            results.append((blue_white_result, "open_regions_image"))
        if blue_red_result is not None:
            results.append((blue_red_result, "blue_red_image"))
        if red_white_result is not None:
            results.append((red_white_result, "red_white_image"))

        for img, _ in results:
            img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            self.image_processed_pub.publish(img_msg)
        
        return results

    def process_and_save(self, mask_primary, mask_check, color_name_primary, color_name_check, processed_images):
        if self.sleep_flag == 1 :
            rospy.sleep(15)
        rospy.loginfo("processing and saving")
        masked_image_primary = cv2.bitwise_and(processed_images, processed_images, mask=mask_primary)
        gray_masked_image_primary = cv2.cvtColor(masked_image_primary, cv2.COLOR_BGR2GRAY)
        non_zero_points_primary = np.column_stack(np.where(gray_masked_image_primary > 0))
        
        #cv2.imshow("edf", masked_image_primary)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        
        
        if non_zero_points_primary.shape[0] == 0:
            # rospy.loginfo(f"No valid points found for {color_name_primary}.")
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
        print("saved") 
        

        return binary_mask, final_result

    def process_second_code(self, blue_white_image):
        if self.sleep_flag == 1 : 
            rospy.sleep(15)
        hsv_image = cv2.cvtColor(blue_white_image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([100, 150, 50])
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
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

        depth_mask = (self.depth_image > self.low_bound) & (self.depth_image < self.upper_bound)
        depth_mask = depth_mask.astype(np.uint8) * 255

        mask_blue = cv2.bitwise_and(mask_blue, mask_blue, mask=depth_mask)
        mask_red = cv2.bitwise_and(mask_red, mask_red, mask=depth_mask)

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
        rospy.loginfo("second code processed ")
        final_result = cv2.bitwise_and(blue_white_image, blue_white_image, mask=binary_mask)
        return final_result

    def process_and_send_to_ml_model(self, image):
        if self.sleep_flag == 1 : 
            rospy.sleep(15)
        rospy.loginfo("sending to model")
        cropped_image = self.crop_non_black(image)
        resize_factor = 56 
        resized_image = cv2.resize(cropped_image, (resize_factor, resize_factor))
        img_msg = self.bridge.cv2_to_imgmsg(resized_image, "bgr8")
        self.image_processed_pub.publish(img_msg)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        input_tensor = transform(resized_image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor.cuda())
            
            max_output = torch.max(output)
            print(max_output)
            if max_output > 0.5: # Tunable
                _, predicted = torch.max(output, 1)
                class_names = ["dead_end", "forward", "left", "no_entry", "right", "stop"]
                
                class_name = class_names[predicted.item()]
                print(class_name)
                self.ml_processed.publish(class_name)
                rospy.loginfo("ML model output: {}".format(class_name))
                self.sign_array.append(class_name)
                rospy.loginfo("node going sleep")
                self.nothing_pub.publish(False)
                rospy.sleep(15)
                self.sleep_flag = 1 
            else:
                self.sleep_flag = 0 
                self.flag = 0 
                self.nothing_pub.publish(True)
                # self.ml_processed.publish ("Wrong sign detected")
            # print("sending")

    def crop_non_black(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped_image = image[y:y + h, x:x + w]
            return cropped_image
        else:
            # self.nothing_pub.publish(True)
            return image  # Return the original image if no non-black pixels are found


    # def sign_calc(self):
    #     # start_time = time.time()
    #     # print("sign cal starting")
    #     # print(len(self.sign_array))

    #     if len(self.sign_array) >= 5:
    #         self.test_list = self.sign_array[:5]
    #         counter = sum(1 for sign in self.test_list if sign == "left")

    #         for i in range(5):
    #             if self.test_list[i] != "left":
    #                 self.ult_list.append(self.test_list[i])
    #             else : 
    #                 counter += 1 
    #             # print(self.test_list)
    #             if counter == 5: 
    #                 self.sign = "left"
    #             else:
    #                 self.sign = self.mod_signs(self.ult_list)
    #                 if self.sign == "deadend" or self.sign == "no entry":
    #                         for i in range(5):
    #                             if self.ult_list[i] == "right" or self.ult_list[i] == "forward" : 
    #                                 self.sign = self.test_list[i]
    #                                 break 
    #                             else:
    #                                 self.sign = "left"
           
    #         self.ml_processed.publish(String(self.sign))
    #         self.publish_sign()
            
    #         end_time = time.time()
    #         # print(end_time-start_time)

    #         print(self.sign)
    #         self.sign_array = self.sign_array[5:]

    # def publish_sign(self):
        
    #     if self.sign == "forward":
    #         print("DM getting Forward")
    #         self.turn_radius_pub.publish(0)
    #     elif self.sign == "right":
    #         print("DM getting right")
    #         self.turn_radius_pub.publish(-0.30)
    #     elif self.sign == "left":
    #         print("DM getting left")
    #         self.turn_radius_pub.publish(0.45)
    #     elif self.sign == "deadend":
    #         print("DM getting deadend")
    #         self.turn_radius_pub.publish(0.45)
    #     elif self.sign == "no_entry":
    #         print("DM getting no_entry")
    #         self.turn_radius_pub.publish()  # Not publishing any value here
    #     print("sign callback working")



# if __name__ == "__main__":
#     rospy.init_node('image_processor_node', anonymous=True)
#     processor = ImageProcessor(camera_topic='/camera/color/image_rect_color', depth_topic='/camera/depth/image_rect_raw', model_path="/home/sedrica/ws/src/cv_only/src/xception_trained_model_1.pth")
#     img_process = ImageProcessor()
#     img_process.sign_calc()
#     rospy.spin()



if __name__ == "__main__":
    rospy.init_node('image_processor_node', anonymous=True)
    start_time = time.time()
    
    # rospy.loginfo("start hoogaya")
    
    processor = ImageProcessor(
            camera_topic='/camera/color/image_raw',
            depth_topic='/camera/aligned_depth_to_color/image_raw',
            model_path="/home/nvidia/Fira_Robo_World_Cup/src/cv_only/src/xception_trained_model_1.pth"
        )
    end_time=time.time()
    rospy.spin()
    # while not rospy.is_shutdown():
       
    #     processor.sign_calc() 
    #     print(end_time-start_time)
      

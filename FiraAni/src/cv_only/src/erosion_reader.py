#!/usr/bin/env python3

import cv2
import numpy as np
import signal
import sys

class ErosionReader:
    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGINT, self.signal_handler)
        self.read()

    def signal_handler(self, sig, frame):
        self.should_exit = True
        print("Interrupted. Exiting...")

    def read(self):
        while not self.should_exit:
            try:
                binary = cv2.imread("binary_erosion.jpg")
                otsu = cv2.imread("otsu_erosion.jpg")
                lane = cv2.imread("lane_image.jpg")

                # Assuming all images should have the same dimensions
                height, width = binary.shape[:2]

                # Resize other images to match the first image's dimensions if they differ
                if lane.shape[:2] != (height, width):
                    lane = cv2.resize(lane, (width, height))
                if otsu.shape[:2] != (height, width):
                    otsu = cv2.resize(otsu, (width, height))

                # Create a green boundary (a column of green pixels)
                green_boundary = np.full((height, 10, 3), (0, 255, 0), dtype=np.uint8)

                # Stack images with green boundaries in between
                concatenated_image = np.hstack((binary, green_boundary, lane, green_boundary, otsu))

                # Resize the concatenated image to fit within half the screen width
                screen_width = 1280  # Adjust this based on your screen size
                scale_factor = screen_width / concatenated_image.shape[1]
                new_width = int(screen_width / 2)
                new_height = int(concatenated_image.shape[0] * (new_width / concatenated_image.shape[1]))
                resized_image = cv2.resize(concatenated_image, (new_width, new_height))

                # Display the resized concatenated image
                cv2.imshow("Concatenated Image", resized_image)

                if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
                    self.should_exit = True

            except Exception as e:
                print(f"An error occurred: {e}")
                self.should_exit = True

        cv2.destroyAllWindows()
        sys.exit()

if __name__ == '__main__':
    obj = ErosionReader()

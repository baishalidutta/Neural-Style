__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# import the necessary packages
import cv2


class MeanPreprocessor:
    def __init__(self, r_mean, g_mean, b_mean):
        # store the Red, Green, and Blue channel averages across a
        # training set
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean

    def preprocess(self, image):
        # split the image into its respective Red, Green, and Blue
        # channels
        (B, G, R) = cv2.split(image.astype("float32"))

        # subtract the means for each channel
        R -= self.r_mean
        G -= self.g_mean
        B -= self.b_mean

        # merge the channels back together and return the image
        return cv2.merge([B, G, R])

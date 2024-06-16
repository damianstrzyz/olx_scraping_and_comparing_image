from matplotlib import pyplot as plt
import numpy as np
import cv2
from typing import List
import os
import imutils


def calculate_matches(des1: List[cv2.KeyPoint], des2: List[cv2.KeyPoint]):
    """
    does a matching algorithm to match if keypoints 1 and 2 are similar
    @param des1: a numpy array of floats that are the descriptors of the keypoints
    @param des2: a numpy array of floats that are the descriptors of the keypoints
    @return:
    """
    # bf matcher with default params
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    topResults = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            topResults.append([m])

    return topResults


def compare_images_kaze():
    cwd = os.getcwd()
    target = os.path.join(cwd, './szukanie_olx/dane', 'sample.png')
    images_list = os.listdir('./szukanie_olx/opencv_images')
    for image in images_list:
        # get my 2 images
        img2 = cv2.imread(target)
        img1 = cv2.imread(os.path.join(cwd, './szukanie_olx/opencv_images', image))
        for i in range(0, 360, int(360 / 1)):
            # rotate my image by i
            img_target_rotation = imutils.rotate_bound(img2, i)

            # Initiate KAZE object with default values
            kaze = cv2.KAZE_create()
            kp1, des1 = kaze.detectAndCompute(img1, None)
            kp2, des2 = kaze.detectAndCompute(img2, None)
            matches = calculate_matches(des1, des2)

            try:
                score = 100 * (len(matches) / min(len(kp1), len(kp2)))
            except ZeroDivisionError:
                score = 0
            print(image, score)
            img3 = cv2.drawMatchesKnn(img1, kp1, img_target_rotation, kp2, matches,
                                      None, flags=2)
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            plt.imshow(img3)
            plt.show()
            plt.clf()


if __name__ == '__main__':
    compare_images_kaze()

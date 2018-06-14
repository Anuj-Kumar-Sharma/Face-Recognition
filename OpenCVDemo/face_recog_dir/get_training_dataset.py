import os
import cv2
import numpy as np


def collect_dataset():
    images = []
    labels = []
    labels_dict = {}
    people = [person for person in os.listdir("people/")]

    for i, person in enumerate(people):
        labels_dict[i] = person

        for image in os.listdir("people/" + person):
            images.append(cv2.imread("people/" + person + "/" + image, 0))
            labels.append(i)

    return images, np.array(labels), labels_dict




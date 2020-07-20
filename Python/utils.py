import os
import fnmatch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from seaborn import heatmap

"""""
This module contains utilities for evaluation & visualization.
"""""


# Displays an image
def show_image(img):
    cv2.imshow('img.png', img)
    cv2.waitKey(0)


# Obtains all images in path with a specific format
def dir_files(img_path, img_ext, person, dataset):
    images = fnmatch.filter(sorted(os.listdir(img_path)), img_ext)
    person_images = [name for name in images if int(name[5:8]) == person]
    img_names = []
    if dataset == 'training':
        img_names = [name for name in person_images if int(name[9:11]) <= 3]
    elif dataset == 'validation':
        img_names = [name for name in person_images if int(name[9:11]) == 4]
    elif dataset == 'testing':
        img_names = [name for name in person_images if int(name[9:11]) >= 5]
    return img_names


# Calculates accuracy & displays confusion matrix
def accuracy(ys, y, st):
    print(st)
    if y.shape[1] > y.shape[0]:
        y = y.transpose()
        ys = ys.transpose()
    if y.shape[1] > 1:
        d = np.argmax(y, axis=1)
        ds = np.argmax(ys, axis=1)
    else:
        d = y
        ds = ys
    c = confusion_matrix(d, ds)
    acc = accuracy_score(d, ds)
    # print('Confusion Matrix:')
    # print(c)
    print(f'--> Accuracy: {acc * 100}%')
    # print()
    nm = c.shape[0]
    plt.figure(figsize=(14, 10))
    heatmap(c, annot=True, fmt="d", cmap="YlGnBu")
    plt.xlim(0, nm)
    plt.ylim(nm, 0)
    plt.title(st, fontsize=14)
    plt.show()
    return acc

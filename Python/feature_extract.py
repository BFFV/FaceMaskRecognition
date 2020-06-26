import numpy as np
import cv2
from pybalu.feature_extraction import lbp_features
from utils import show_image, dir_files

"""""
This module contains functions for feature extraction.
"""""


# Reads an image
def get_image(path, show=False):
    img = cv2.imread(path)
    if show:
        show_image(img)
    return img


# Extracts features from an image
def extract_features_img(image, selected):
    img = get_image(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:128, 32:]  # Processed Image
    features = np.array([])
    if 'lbp' in selected:  # Local Binary Patterns
        lbp = lbp_features(gray, hdiv=8, vdiv=8, mapping='nri_uniform')
        features = np.concatenate((features, lbp))
    return features


# Extracts features for all images in dir_path
def extract_features(dir_path, fmt, selected, person, dataset):
    st = '*.' + fmt
    img_names = dir_files(dir_path + '/', st, person, dataset)
    n = len(img_names)
    data = []
    for i in range(n):
        img_path = img_names[i]
        features = extract_features_img(dir_path + '/' + img_path, selected)
        if not i:
            m = features.shape[0]
            data = np.zeros((n, m))
        data[i] = features
    return data

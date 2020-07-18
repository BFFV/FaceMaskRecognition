import numpy as np
import cv2
from skimage.feature import hog
from pybalu.feature_extraction import lbp_features, haralick_features, \
    gabor_features
from utils import show_image, dir_files

"""""
This module contains functions for feature extraction.
"""""


# Read and process an image
def process_image(path, show=False):
    img = cv2.imread(path)
    if show:
        show_image(img)
    # Gaussian Blur
    # filtered_img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    filtered_img = img
    # show_image(filtered_img)
    return filtered_img[:100, 32:200]


# Extracts features from an image
def extract_features_img(image, selected):
    img = process_image(image)  # Processed Image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gray Scale
    features = np.array([])
    if 'lbp' in selected:  # Local Binary Patterns
        lbp_gray = lbp_features(gray, hdiv=8, vdiv=8, mapping='nri_uniform')
        lbp_rgb = lbp_features(img[:, :, 0],
                               hdiv=8, vdiv=8, mapping='nri_uniform')
        lbp = np.concatenate((lbp_gray, lbp_rgb))
        features = np.concatenate((features, lbp))
    if 'hog' in selected:  # Histogram of Gradients
        hog_features = hog(gray, orientations=16, pixels_per_cell=(16, 16),
                           cells_per_block=(4, 4))
        features = np.concatenate((features, hog_features))
    if 'haralick' in selected:  # Haralick Textures
        haralick = haralick_features(gray, distance=3)
        features = np.concatenate((features, haralick))
    if 'gabor' in selected:  # Gabor Features
        gabor = gabor_features(gray, rotations=8, dilations=8)
        features = np.concatenate((features, gabor))
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

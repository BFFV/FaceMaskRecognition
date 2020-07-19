import numpy as np
import cv2
from skimage.feature import hog
from mahotas.features import zernike_moments
from pybalu.feature_extraction import lbp_features, haralick_features, \
    gabor_features
from utils import show_image, dir_files

"""""
This module contains functions for feature extraction.
"""""


# Read and process an image
def process_image(path, show=False):
    img = cv2.imread(path)
    # Cut the background from the image
    cut_img = img[:100, 35:220]
    # Gray Scale
    gray = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
    # Blur the image
    blur_img = cv2.GaussianBlur(cut_img, (1, 1), cv2.BORDER_DEFAULT)
    blur_gray = cv2.GaussianBlur(gray, (1, 1), cv2.BORDER_DEFAULT)
    # Equalize the lighting
    # b, g, r = cv2.split(cut_img)
    # equalized_b = cv2.equalizeHist(b)
    # equalized_g = cv2.equalizeHist(g)
    # equalized_r = cv2.equalizeHist(r)
    # equalized_img = cv2.merge((equalized_b, equalized_g, equalized_r))
    # equalized_gray = cv2.equalizeHist(gray)
    if show:
        show_image(blur_img)
    return blur_img, blur_gray


# Extracts features from an image
def extract_features_img(image, selected):
    img, gray = process_image(image)  # Processed Image
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
    if 'zernike' in selected:  # Zernike Moments
        zer_moments = zernike_moments(gray, 80)
        features = np.concatenate((features, zer_moments))
    if 'sift' in selected:  # SIFT Descriptors
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            descriptors = np.full(1280, 50, dtype=int)
        elif descriptors.shape[0] < 10:
            default = np.full(128 * (10 - descriptors.shape[0]), 50, dtype=int)
            descriptors = np.concatenate((descriptors.flatten(), default))
        else:
            descriptors = descriptors[:10].flatten()
        features = np.concatenate((features, descriptors))
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

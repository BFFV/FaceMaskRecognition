import numpy as np
from pybalu.feature_selection import clean
from pybalu.feature_transformation import normalize
from feature_extract import extract_features
from feature_select_transform import select_features, transform_features
from classification import classify
from utils import accuracy

# Train or load from data file

load = False
train_data = 'train_data.npy'  # Training Set Data
train_classes = 'train_classes.npy'  # Training Set Classes
validation_data = 'validation_data.npy'  # Validation Set Data
validation_classes = 'validation_classes.npy'  # Validation Set Classes

# Image Set
img_set = 'A'
set_dict = {'A': 17, 'B': 41, 'C': 101, 'D': 167}

# Features to extract (gabor, haralick, hog, lbp)
selected_features = ['lbp']

# Selection/Transformation steps (sfs, mutual_info, pca)
# sfs => n_features: int, method: ('fisher', 'sp100')
# mutual_info => n_features: int, n_neighbors: int
# pca => n_components: int, energy: float in [0,1]

strategy_1 = [  # SFS + MI
    ['pca', {'n_components': 2500}],
    ['mutual_info', {'n_features': 800, 'n_neighbors': 3}]]
strategy_2 = [  # SFS + PCA
    ['pca', {'n_components': 2500}]]
strategy_3 = [  # Best Combination
    ['sfs', {'n_features': 24, 'method': 'fisher'}]]

processing_strategy = []

# Classifier to use (knn, dmin, lda, svm, nn, random_forest, adaboost)
# knn => n_neighbors: int, weights: ('uniform', 'distance')
# dmin => no params
# lda => no params
# svm => C: float, kernel: ('linear', 'poly', 'rbf', 'sigmoid')
# nn => hidden_layer_sizes: (size_1, size_2),
# activation: ('identity', 'logistic', 'tanh', 'relu'), max_iter: int
# random_forest => n_estimators: int, criterion: ('gini', 'entropy'),
# max_depth: int/None
# adaboost => n_estimators: int, learning_rate: float

classifier_1 = ['svm', {'C': 1, 'kernel': 'linear'}]
classifier_2 = ['nn', {'hidden_layer_sizes': (100,),
                       'activation': 'logistic', 'max_iter': 2000,
                       'random_state': 1}]
classifier_3 = ['random_forest', {'n_estimators': 1200, 'criterion': 'entropy',
                                  'max_depth': None, 'random_state': 1}]
# classifier_4 = ['knn', {'n_neighbors': 3, 'weights': 'distance'}]
# classifier_5 = ['dmin', {}]
# classifier_6 = ['lda', {}]
# classifier_7 = ['adaboost', {'n_estimators': 1000, 'learning_rate': 1}]

classifier = classifier_2

# Training Set
print('Training...')
if load:  # Load extracted features from data file
    X_train = np.load(train_data)
    d_train = np.load(train_classes)
else:  # Extract features from all images & save the data
    X_train = np.array([])
    d_train = np.array([])
    for person in range(1, set_dict[img_set]):
        current_features = extract_features(
            '../FaceMask166', 'jpg', selected_features, person, 'training')
        current_classes = np.full(3, person, dtype=int)
        if not X_train.shape[0]:
            X_train = np.array(current_features)
            d_train = np.array(current_classes)
        else:
            X_train = np.concatenate((X_train, current_features), axis=0)
            d_train = np.concatenate((d_train, current_classes), axis=0)
    np.save('train_data.npy', X_train)
    np.save('train_classes.npy', d_train)
print(f'Original Extracted Features: {X_train.shape[1]} ({X_train.shape[0]} '
      f'samples)')
print('Cleaning...')
s_clean = clean(X_train, show=True)
X_train_clean = X_train[:, s_clean]
print(f'           cleaned features: {X_train_clean.shape[1]} '
      f'({X_train_clean.shape[0]} samples)')
print('Normalizing...')
X_train_norm, a, b = normalize(X_train_clean)
print(f'        normalized features: {X_train_norm.shape[1]} '
      f'({X_train_norm.shape[0]} samples)')
print('Selecting/Transforming Features...')
X_train_final = X_train_norm
for index, step in enumerate(processing_strategy):
    if step[0] in ['sfs', 'mutual_info']:  # Selection
        step[1]['n_features'] = \
            min(X_train_final.shape[1], step[1]['n_features'])
        output = select_features(X_train_final, d_train, step[0], step[1])
        X_train_final = X_train_final[:, output]
        processing_strategy[index].append(output)
    else:  # Transformation
        step[1]['n_components'] = \
            min(X_train_final.shape[1], step[1]['n_components'])
        output = transform_features(X_train_final, step[0], step[1])
        X_train_final = output[0]
        processing_strategy[index].append(output[1])
print(f'          selected/transformed features: {X_train_final.shape[1]} '
      f'({X_train_final.shape[0]} samples)')

# Validation Set
print('Validating...')
if load:  # Load extracted features from data file
    X_validate = np.load(validation_data)
    d_validate = np.load(validation_classes)
else:  # Extract features from all images & save the data
    X_validate = np.array([])
    d_validate = np.array([])
    for person in range(1, set_dict[img_set]):
        current_features = extract_features(
            '../FaceMask166', 'jpg', selected_features, person, 'validation')
        current_classes = np.full([1, 1], person, dtype=int)
        if not X_validate.shape[0]:
            X_validate = np.array(current_features)
            d_validate = np.array(current_classes)
        else:
            X_validate = np.concatenate((X_validate, current_features), axis=0)
            d_validate = np.concatenate((d_validate, current_classes), axis=0)
    np.save('validation_data.npy', X_validate)
    np.save('validation_classes.npy', d_validate)
print(f'Original Extracted Features: {X_validate.shape[1]} '
      f'({X_validate.shape[0]} samples)')
print('Cleaning...')
X_validate_clean = X_validate[:, s_clean]
print('Normalizing...')
X_validate_norm = X_validate_clean * a + b
print('Selecting/Transforming Features...')
X_validate_final = X_validate_norm
for index, step in enumerate(processing_strategy):
    if step[0] in ['sfs', 'mutual_info']:  # Selection
        selected = step[2]
        X_validate_final = X_validate_final[:, selected]
    elif step[0] == 'pca':  # PCA
        params = step[2]
        X_validate_final = np.matmul(
            X_validate_final - params['Xm'], params['A'])
print(f'    clean+norm+selected/transformed features:'
      f' {X_validate_final.shape[1]} ({X_validate_final.shape[0]} samples)')

# Testing Set
print('Testing...')
X_test = np.array([])
d_test = np.array([])
for person in range(1, set_dict[img_set]):
    current_features = extract_features(
        '../FaceMask166', 'jpg', selected_features, person, 'testing')
    current_classes = np.full([2, 1], person, dtype=int)
    if not X_test.shape[0]:
        X_test = np.array(current_features)
        d_test = np.array(current_classes)
    else:
        X_test = np.concatenate((X_test, current_features), axis=0)
        d_test = np.concatenate((d_test, current_classes), axis=0)
print(f'Original Extracted Features: {X_test.shape[1]} ({X_test.shape[0]} '
      f'samples)')
print('Cleaning...')
X_test_clean = X_test[:, s_clean]
print('Normalizing...')
X_test_norm = X_test_clean * a + b
print('Selecting/Transforming Features...')
X_test_final = X_test_norm
for index, step in enumerate(processing_strategy):
    if step[0] in ['sfs', 'mutual_info']:  # Selection
        selected = step[2]
        X_test_final = X_test_final[:, selected]
    elif step[0] == 'pca':  # PCA
        params = step[2]
        X_test_final = np.matmul(
            X_test_final - params['Xm'], params['A'])
print(f'    clean+norm+selected/transformed features:'
      f' {X_test_final.shape[1]} ({X_test_final.shape[0]} samples)')

# Classification
print('Classifying...\n')
validation_results = classify(
    X_train_final, X_validate_final, d_train, classifier[0], classifier[1])
accuracy(validation_results, d_validate, 'Validation')
testing_results = classify(
    X_train_final, X_test_final, d_train, classifier[0], classifier[1])
accuracy(testing_results, d_test, 'Person Classification')
print('Finished!')

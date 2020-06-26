import numpy as np
from feature_extract import extract_features
from classification import classify
from utils import accuracy


# Features to extract (lbp)
selected_features = ['lbp']

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

classifier_1 = ['svm', {'C': 1, 'kernel': 'rbf'}]
classifier_2 = ['nn', {'hidden_layer_sizes': (200, 200), 'activation': 'relu',
                       'max_iter': 2000, 'random_state': 1}]
classifier_3 = ['random_forest', {'n_estimators': 1200, 'criterion': 'entropy',
                                  'max_depth': None, 'random_state': 1}]
# classifier_4 = ['knn', {'n_neighbors': 3, 'weights': 'distance'}]
# classifier_5 = ['dmin', {}]
# classifier_6 = ['lda', {}]
# classifier_7 = ['adaboost', {'n_estimators': 1000, 'learning_rate': 1}]

classifier = classifier_3

# Training Set
print('Training...')
X_train = np.array([])
d_train = np.array([])
for person in range(1, 17):
    current_features = extract_features(
        'data', 'jpg', selected_features, person, 'training')
    current_classes = np.full(3, person, dtype=int)
    if not X_train.shape[0]:
        X_train = np.array(current_features)
        d_train = np.array(current_classes)
    else:
        X_train = np.concatenate((X_train, current_features), axis=0)
        d_train = np.concatenate((d_train, current_classes), axis=0)
print(f'Original Extracted Features: {X_train.shape[1]} ({X_train.shape[0]} '
      f'samples)')

# Validation Set
print('Validating...')
X_validate = np.array([])
d_validate = np.array([])
for person in range(1, 17):
    current_features = extract_features(
        'data', 'jpg', selected_features, person, 'validation')
    current_classes = np.full([1, 1], person, dtype=int)
    if not X_validate.shape[0]:
        X_validate = np.array(current_features)
        d_validate = np.array(current_classes)
    else:
        X_validate = np.concatenate((X_validate, current_features), axis=0)
        d_validate = np.concatenate((d_validate, current_classes), axis=0)
print(f'Original Extracted Features: {X_validate.shape[1]} '
      f'({X_validate.shape[0]} samples)')

# Testing Set
print('Testing...')
X_test = np.array([])
d_test = np.array([])
for person in range(1, 17):
    current_features = extract_features(
        'data', 'jpg', selected_features, person, 'testing')
    current_classes = np.full([2, 1], person, dtype=int)
    if not X_test.shape[0]:
        X_test = np.array(current_features)
        d_test = np.array(current_classes)
    else:
        X_test = np.concatenate((X_test, current_features), axis=0)
        d_test = np.concatenate((d_test, current_classes), axis=0)
print(f'Original Extracted Features: {X_test.shape[1]} ({X_test.shape[0]} '
      f'samples)')

# Classification
print('Classifying...\n')
validation_results = classify(
    X_train, X_validate, d_train, classifier[0], classifier[1])
accuracy(validation_results, d_validate, 'Validation')
testing_results = classify(
    X_train, X_test, d_train, classifier[0], classifier[1])
accuracy(testing_results, d_test, 'Person Classification')
print('Finished!')

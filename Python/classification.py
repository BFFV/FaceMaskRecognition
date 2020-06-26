from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

"""""
This module contains functions for image classification.
"""""


# Classifies images
def classify(train_data, test_data, classes, classifier, kwargs):
    selected = []  # Classifier
    if classifier == 'knn':  # k-Nearest Neighbors
        selected = KNeighborsClassifier(**kwargs)
    elif classifier == 'dmin':  # Minimal Distance
        selected = NearestCentroid()
    elif classifier == 'lda':  # Linear Discriminant Analysis
        selected = LinearDiscriminantAnalysis()
    elif classifier == 'svm':  # Support Vector Machine
        selected = SVC(**kwargs)
    elif classifier == 'nn':  # Neural Network
        selected = MLPClassifier(**kwargs)
    elif classifier == 'random_forest':  # Random Forest
        selected = RandomForestClassifier(**kwargs)
    elif classifier == 'adaboost':  # AdaBoost
        selected = AdaBoostClassifier(**kwargs)
    selected.fit(train_data, classes)
    return selected.predict(test_data)

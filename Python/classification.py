from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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
    elif classifier == 'linear_svm':  # Linear Support Vector Machine
        selected = LinearSVC(**kwargs)
    elif classifier == 'nn':  # Neural Network
        selected = MLPClassifier(**kwargs)
    elif classifier == 'random_forest':  # Random Forest
        selected = RandomForestClassifier(**kwargs)
    elif classifier == 'adaboost':  # AdaBoost
        selected = AdaBoostClassifier(**kwargs)
    elif classifier == 'log_reg':  # Logistic Regression
        selected = LogisticRegression(**kwargs)
    selected.fit(train_data, classes)
    return selected.predict(test_data)

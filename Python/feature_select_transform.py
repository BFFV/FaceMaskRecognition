from pybalu.feature_selection import sfs
from pybalu.feature_transformation import pca
from sklearn.feature_selection import SelectKBest, mutual_info_classif, \
    f_classif

"""""
This module contains functions for feature selection/transformation.
"""""


# Selects features
def select_features(train_data, classes, operation, kwargs):
    if operation == 'sfs':  # Sequential Forward Selection
        return sfs(train_data, classes, **kwargs)
    elif operation == 'mutual_info':  # Mutual Information
        m_info = SelectKBest(mutual_info_classif, k=kwargs['n_features'])
        m_info.fit(train_data, classes)
        return m_info.get_support(indices=True)
    elif operation == 'anova_f':  # ANOVA F-values
        anova_f = SelectKBest(f_classif, k=kwargs['n_features'])
        anova_f.fit(train_data, classes)
        return anova_f.get_support(indices=True)


# Transforms features
def transform_features(train_data, operation, kwargs):
    if operation == 'pca':  # Principal Component Analysis
        features, _, a, xm, _ = pca(train_data, **kwargs)
        return features, {'A': a, 'Xm': xm}

# -*- coding: utf-8 -*-

from graalpy.utils.majority_vote import ClassifiersGenerator, DecisionStumpClassifier

import numpy as np

from joblib import Parallel, delayed

import os

class CustomStumpsClassifiersGenerator(ClassifiersGenerator):
    """Decision Stump Voters transformer.

    Parameters
    ----------
    n_stumps_per_attribute : int, optional
        Determines how many decision stumps will be created for each attribute. Defaults to 10.
        No stumps will be created for attributes with only one possible value.
    self_complemented : bool, optional
        Whether or not a binary complement voter must be generated for each voter. Defaults to False.

    """
    def __init__(self, pickle_path):
        super(CustomStumpsClassifiersGenerator, self).__init__()
        self.pickle_path = pickle_path

    def fit(self, X, y, classes_weights=None, n_jobs=-1):
        """Fits Decision Stump voters on a training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data on which to base the voters.
        y : ndarray of shape (n_labeled_samples,), optional
            Only used to ensure that we are in the binary classification setting.

        Returns
        -------
        self

        """
        X_args_sorted = np.argsort(X, axis=0)

        # if classes_weights is not None:
        #     y = np.multiply(y, classes_weights)

        liste = Parallel(n_jobs=n_jobs)(delayed(get_best_votant_feature)(X[:, i], y, X_args_sorted[:, i], i) for i in np.arange(X.shape[1]))

        self.estimators_ = np.asarray([liste[i][0] for i in np.arange(len(liste)) if liste[i][1] > 0.])

        np.save(self.pickle_path + "estimators", self.estimators_)

        return self


def get_best_votant_feature(X_i, y, sorted_args_i, i):
    # Obtenir le feature i et le classer en ordre. Appliquer le même ordre à y pour avoir la correspondance.
    X_i_sorted = X_i[sorted_args_i]
    y_i_sorted = y[sorted_args_i]

    '''
    Obtention des votants possibles. Une coupe est faite entre deux valeurs (distinctes) consécutives.
    '''
    X_i_unique = np.unique(X_i_sorted)
    k = X_i_unique.shape[0]

    if k == 1:  # On avait une seule valeur pour ce feature, on ne veut rien faire
        return [0., -1]

    X_i_stumps = (X_i_unique[:-1] + X_i_unique[1:]) * 0.5
    arrays = np.asarray([X_i_sorted - stump for stump in X_i_stumps])
    votants_i_neg_first = np.sign(np.vstack(arrays))
    votants_i_pos_first = -votants_i_neg_first
    votants = np.concatenate((votants_i_pos_first, votants_i_neg_first), axis=0)


    '''
    Calcul de l'accuracy
    '''
    correct = np.multiply(votants, y_i_sorted)
    accuracy = np.apply_along_axis(lambda line: line[line > 0].sum() / X_i_sorted.shape[0], 1, correct)

    '''
    Obtention du votant avec la meilleure accuracy
    '''
    best_votant_idx = np.argmax(accuracy)
    idx = best_votant_idx % X_i_stumps.shape[0]
    direction = best_votant_idx / X_i_stumps.shape[0] * 2 - 1

    votant = DecisionStumpClassifier(i, X_i_stumps[idx], direction).fit(X_i, y)

    acc = accuracy[best_votant_idx]

    print(str(i) + " : " + str(acc))

    return [votant, acc]
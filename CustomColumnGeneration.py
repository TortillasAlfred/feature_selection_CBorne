# -*- coding: utf-8 -*-
"""Base class for all column-generation-based algorithms.

"""
from __future__ import division, print_function, absolute_import
from future.utils import iteritems
import logging
import scipy
from copy import deepcopy
from collections import defaultdict
import numpy as np
import numpy.ma as ma
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from graalpy.utils.majority_vote import StumpsClassifiersGenerator
from graalpy.results_dataframe import ResultsDataFrame
from graalpy.metrics import zero_one_loss, zero_one_loss_per_example
from graalpy.utils.misc import sign

from tests import pickle_path

class ColumnGenerationClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epsilon=1e-06, n_max_iterations=None, estimators_generator=None, dual_constraint_rhs=0, save_iteration_as_hyperparameter_each=None,
                        classes_weights=None):
        self.epsilon = epsilon
        self.n_max_iterations = n_max_iterations
        self.estimators_generator = estimators_generator
        self.dual_constraint_rhs = dual_constraint_rhs
        self.save_iteration_as_hyperparameter_each = save_iteration_as_hyperparameter_each
        self.classes_weights = classes_weights # Hack CFS

    def fit(self, X, y, from_pickle=False):
        if scipy.sparse.issparse(X):
            logging.info('Converting to dense matrix.')
            X = np.array(X.todense())

        if self.estimators_generator is None:
            self.estimators_generator = StumpsClassifiersGenerator(n_stumps_per_attribute=10, self_complemented=True)

        # Hack CFS
        if from_pickle:
            self.estimators_generator.estimators_ = np.load(pickle_path + "estimators")
        else :
            self.estimators_generator.fit(X, y, classes_weights=self.classes_weights)

        classification_matrix = self._binary_classification_matrix(X)

        self.chosen_columns_ = []
        self.infos_per_iteration_ = defaultdict(list)

        m, n = classification_matrix.shape
        self.n_total_hypotheses_ = n

        y_kernel_matrix = np.multiply(y.reshape((len(y), 1)), classification_matrix)

        # Hack CFS (2eme version)
        if self.classes_weights is not None :
            y_kernel_matrix = np.multiply(y_kernel_matrix.T, self.classes_weights).T

        # Initialization
        alpha = self._initialize_alphas(m)
        w = None
        self.collected_weight_vectors_ = {}
        self.collected_dual_constraint_violations_ = {}

        for k in range(min(n, self.n_max_iterations if self.n_max_iterations is not None else np.inf)):
            print(k)

            # Find worst weak hypothesis given alpha.
            h_values = ma.array(np.squeeze(np.array(alpha.T.dot(y_kernel_matrix).T)), fill_value=-np.inf)
            h_values[self.chosen_columns_] = ma.masked
            worst_h_index = ma.argmax(h_values)
            logging.info("Adding voter {} to the columns, value = {}".format(worst_h_index, h_values[worst_h_index]))

            # Check for optimal solution. We ensure at least one complete iteration is done as the initialization
            # values might provide a degenerate initial solution.
            if h_values[worst_h_index] <= self.dual_constraint_rhs + self.epsilon and len(self.chosen_columns_) > 0:
                break

            # Append the weak hypothesis.
            self.chosen_columns_.append(worst_h_index)

            # Solve restricted master for new costs.
            w, alpha = self._restricted_master_problem(y_kernel_matrix[:, self.chosen_columns_], previous_w=w, previous_alpha=alpha)

            # We collect iteration information for later evaluation.
            if self.save_iteration_as_hyperparameter_each is not None:
                if (k + 1) % self.save_iteration_as_hyperparameter_each == 0:
                    self.collected_weight_vectors_[k] = deepcopy(w)
                    self.collected_dual_constraint_violations_[k] = h_values[worst_h_index] - self.dual_constraint_rhs

        self.weights_ = w
        self.estimators_generator.estimators_ = self.estimators_generator.estimators_[self.chosen_columns_]

        self.learner_info_ = {}
        self.learner_info_.update(n_nonzero_weights=np.sum(np.asarray(self.weights_) > 1e-12))
        self.learner_info_.update(n_generated_columns=len(self.chosen_columns_))

        return self

    def predict(self, X):
        check_is_fitted(self, 'weights_')

        if scipy.sparse.issparse(X):
            logging.warning('Converting sparse matrix to dense matrix.')
            X = np.array(X.todense())

        classification_matrix = self._binary_classification_matrix(X)

        margins = np.squeeze(np.asarray(np.dot(classification_matrix, self.weights_)))
        return np.array([int(x) for x in sign(margins)])

    def _binary_classification_matrix(self, X):
        probas = self._collect_probas(X)
        predicted_labels = np.argmax(probas, axis=2)
        predicted_labels[predicted_labels == 0] = -1
        values = np.max(probas, axis=2)
        return (predicted_labels * values).T

    def _collect_probas(self, X):
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_generator.estimators_])

    def _restricted_master_problem(self, y_kernel_matrix):
        raise NotImplementedError("Restricted master problem not implemented.")

    def _initialize_alphas(self, n_examples):
        raise NotImplementedError("Alpha weights initialization function is not implemented.")

    def evaluate_metrics(self, X, y, metrics_list=None, functions_list=None):
        if metrics_list is None:
            metrics_list = [zero_one_loss, zero_one_loss_per_example]

        if functions_list is None:
            functions_list = []

        # Predict, evaluate metrics.
        classification_matrix = self._binary_classification_matrix(X)
        predictions = sign(classification_matrix.dot(self.weights_))

        if self.save_iteration_as_hyperparameter_each is None:
            metrics_results = {}
            for metric in metrics_list:
                metrics_results[metric.__name__] = metric(y, predictions)

            metrics_dataframe = ResultsDataFrame([metrics_results])
            return metrics_dataframe

        # If we collected iteration informations to add a hyperparameter, we add an index with the hyperparameter name
        # and return a ResultsDataFrame containing one row per hyperparameter value.
        metrics_dataframe = ResultsDataFrame()
        for t, weights in iteritems(self.collected_weight_vectors_):
            predictions = sign(classification_matrix[:, :t + 1].dot(weights))
            metrics_results = {metric.__name__: metric(y, predictions) for metric in metrics_list}
            for function in functions_list:
                metrics_results[function.__name__] = function(classification_matrix[:, :t + 1], y, weights)

            # We add other collected information.
            metrics_results['chosen_columns'] = self.chosen_columns_[t]
            metrics_results['dual_constraint_violation'] = self.collected_dual_constraint_violations_[t]

            metrics_dataframe = metrics_dataframe.append(ResultsDataFrame([metrics_results], index=[t]))

        metrics_dataframe.index.name = 'hp__n_iterations'
        return metrics_dataframe

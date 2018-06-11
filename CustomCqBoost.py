# -*- coding: utf-8 -*-
"""CqBoost algorithm.

Roy, Jean-Francis, Mario Marchand and François Laviolette (2016). "A Column Generation Bound Minimization Approach with PAC-Bayesian Generalization Guarantees."
In: Proceedings of the Nineteenth International Conference on Artificial Intelligence and Statistics (AISTATS), p. 1241-1249.

"""
from __future__ import division, print_function, absolute_import
import logging
import numpy as np
from graalpy.solvers.convex_programming import ConvexProgram
from CustomColumnGeneration import ColumnGenerationClassifier


class CqBoostClassifier(ColumnGenerationClassifier):
    def __init__(self, mu=0.001, epsilon=1e-08, n_max_iterations=None, estimators_generator=None, save_iteration_as_hyperparameter_each=None, classes_weights=None):
        super(CqBoostClassifier, self).__init__(epsilon, n_max_iterations, estimators_generator, dual_constraint_rhs=0,
                                                save_iteration_as_hyperparameter_each=save_iteration_as_hyperparameter_each, classes_weights=classes_weights)
        # TODO: Vérifier la valeur de nu (dual_constraint_rhs) à l'initialisation, mais de toute manière ignorée car
        # on ne peut pas quitter la boucle principale avec seulement un votant.
        self.mu = mu

    def _restricted_master_problem(self, y_kernel_matrix, previous_w=None, previous_alpha=None):
        n_examples, n_hypotheses = y_kernel_matrix.shape

        m_eye = np.eye(n_examples)
        m_ones = np.ones((n_examples, 1))

        qp_a = np.vstack((np.hstack((-y_kernel_matrix, m_eye)),
                          np.hstack((np.ones((1, n_hypotheses)), np.zeros((1, n_examples))))))

        qp_b = np.vstack((np.zeros((n_examples, 1)),
                          np.array([1.0]).reshape((1, 1))))

        # Hack CFS (version 3), ajouter les poids des classes dans qp_g
        qp_g = np.vstack((np.hstack((-np.eye(n_hypotheses), np.zeros((n_hypotheses, n_examples)))),
                          np.hstack((np.zeros((1, n_hypotheses)), - 1.0 / n_examples * m_ones.T * self.classes_weights))))

        qp_h = np.vstack((np.zeros((n_hypotheses, 1)),
                          np.array([-self.mu]).reshape((1, 1))))

        qp = ConvexProgram()
        qp.quadratic_func = 2.0 / n_examples * np.vstack((np.hstack((np.zeros((n_hypotheses, n_hypotheses)), np.zeros((n_hypotheses, n_examples)))),
                                                        np.hstack((np.zeros((n_examples, n_hypotheses)), m_eye))))

        qp.add_equality_constraints(qp_a, qp_b)
        qp.add_inequality_constraints(qp_g, qp_h)

        if previous_w is not None:
            qp.initial_values = np.append(previous_w, [0])

        try:
            solver_result = qp.solve(abstol=1e-10, reltol=1e-10, feastol=1e-10, return_all_information=True)
            w = np.asarray(np.array(solver_result['x']).T[0])[:n_hypotheses]

            # The alphas are the Lagrange multipliers associated with the equality constraints (returned as the y vector in CVXOPT).
            dual_variables = np.asarray(np.array(solver_result['y']).T[0])
            alpha = dual_variables[:n_examples]

            # Set the dual constraint right-hand side to be equal to the last lagrange multiplier (nu).
            # Hack: do not change nu if the QP didn't fully solve...
            if solver_result['dual slack'] <= 1e-8:
                self.dual_constraint_rhs = dual_variables[-1]
                logging.info('Updating dual constraint rhs: {}'.format(self.dual_constraint_rhs))

        except:
            logging.warning('QP Solving failed at iteration {}.'.format(n_hypotheses))
            if previous_w is not None:
                w = np.append(previous_w, [0])
            else:
                w = np.array([1.0 / n_hypotheses] * n_hypotheses)

            if previous_alpha is not None:
                alpha = previous_alpha
            else:
                alpha = self._initialize_alphas(n_examples)

        return w, alpha

    def _initialize_alphas(self, n_examples):
        # Hack CFS (version 1, plus la : initialiser alpha avec class_weights)
        return 1.0 / n_examples * np.ones((n_examples,))

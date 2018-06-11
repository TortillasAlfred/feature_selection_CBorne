# -*- coding: utf-8 -*-

from pyscm import SetCoveringMachineClassifier

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, make_scorer

from CustomCqBoost import CqBoostClassifier

from StumpsGenerator import CustomStumpsClassifiersGenerator

import numpy as np

import cPickle as pickle
import os

pickle_path = os.path.join(os.path.realpath('..'), 'CFS/', 'pickles/')

random_state = 420

algos_tested = dict(pyscm=SetCoveringMachineClassifier(random_state=random_state),
                    svm_lin=SVC(class_weight='balanced', random_state=random_state, kernel='linear'),
                    svm_rbf=SVC(class_weight='balanced', random_state=random_state, kernel='rbf'),
                    decisionTree=DecisionTreeClassifier(class_weight='balanced', random_state=random_state),
                    randomForest=RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=random_state))

params = dict(pyscm=dict(p=[0.1, 0.5, 1., 2., 5., 10.], model_type=['disjunction', 'conjunction'], max_rules=[1, 2, 3, 4, 5]),
              svm_lin=dict(C=np.logspace(-2, 2, 20)),
              svm_rbf=dict(C=np.logspace(-2, 2, 20), gamma=np.logspace(-2, 2, 20)),
              decisionTree=dict(max_depth=[2, 3, 5, 10], min_samples_split=[4, 6, 8]),
              randomForest=dict(max_depth=[2, 3, 5, 10], min_samples_split=[4, 6, 8]))

def wa(y_true, y_pred, weights):
    w_y = np.asarray(y_true, dtype='float64')
    for k in weights.keys():
        w_y[y_true == k] = k * weights[k]
    correct = np.multiply(y_pred, w_y)
    return correct[correct > 0].sum()/correct.shape[0]

def get_classifier_scores(X, y, sample_weights, random_states=[1, 42, 69, 420, 666, 33, 17, 6, 0, 51], cv=True):
    f1_scores_mean = {}
    f1_scores_var = {}
    accuracies_mean = {}
    accuracies_var = {}

    for algo in algos_tested:
        f1_scores_mean_algo = []
        f1_scores_var_algo = []
        accuracies_mean_algo = []
        accuracies_var_algo = []

        print "Begin " + str(algo)

        for x in X:
            f1_temp, accuracies_temp = [], []
            for rs in random_states:
                learner = algos_tested[algo]
                x_train, x_test, y_train, y_test, _, sample_weights_test = train_test_split(x, y, sample_weights, train_size=0.7, random_state=rs)
                if cv:
                    cv_params = params[algo]
                    # Weight_dict est hardcodé avec les poids retournés par determiner_class_weights()
                    weight_dict = {}
                    weight_dict[-1] = 0.594979647218453
                    weight_dict[1] = 3.13214285714286
                    scorer = make_scorer(wa, weights=weight_dict)
                    # Cross validation avec weighted accuracy
                    clf = GridSearchCV(learner, cv_params, verbose=1, n_jobs=-1, scoring=scorer, cv=5)
                else:
                    clf = learner
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                f1_temp.append(f1_score(y_test, y_pred))
                accuracies_temp.append(accuracy_score(y_test, y_pred, sample_weight=sample_weights_test))

            f1_temp = np.asarray(f1_temp)
            f1_scores_mean_algo.append(np.mean(f1_temp, dtype=np.float64))
            f1_scores_var_algo.append(np.var(f1_temp, dtype=np.float64))
            accuracies_temp = np.asarray(accuracies_temp)
            accuracies_mean_algo.append(np.mean(accuracies_temp, dtype=np.float64))
            accuracies_var_algo.append(np.var(accuracies_temp, dtype=np.float64))

        f1_scores_mean[algo] = f1_scores_mean_algo
        f1_scores_var[algo] = f1_scores_var_algo
        accuracies_mean[algo] = accuracies_mean_algo
        accuracies_var[algo] = accuracies_var_algo

    return f1_scores_mean, f1_scores_var, accuracies_mean, accuracies_var

def calculer_similarites_features():
    retained_features_CqBoost = np.load(open(pickle_path + "retained_features_CqBoost.pck"))
    retained_features_RFE = np.load(open(pickle_path + "retained_features_RFE.pck"))
    retained_features_RFE_5000 = np.load(open(pickle_path + "retained_features_RFE_5000.pck"))

    retained_features_CqBoost_sizes, retained_features_RFE_sizes = [], []

    sizes = calculer_sizes()

    for s in sizes:
        retained_features_CqBoost_sizes.append(retained_features_CqBoost[:s])
        retained_features_RFE_sizes.append(retained_features_RFE_5000[retained_features_RFE[:s]])

    percent_identical_features= []
    for i in range(len(sizes)):
        x1 = retained_features_CqBoost_sizes[i]
        x2 = retained_features_RFE_sizes[i]
        percent_identical_features.append(np.sum(np.in1d(x1, x2), dtype='float64') / sizes[i])

    return percent_identical_features

def run_tests_comparison():
    _, y = load_data()
    class_weights = determiner_poids_classes(y)

    sizes = calculer_sizes()

    X_RFE = [np.load(open(pickle_path + "X_" + str(n) + "_RFE.pck", 'rb')) for n in sizes]
    f1_scores_mean_RFE, f1_scores_var_RFE, accuracies_mean_RFE, accuracies_var_RFE = get_classifier_scores(X_RFE, y, sample_weights=class_weights)

    print "Done tests RFE"

    X_CqBoost = [np.load(open(pickle_path + "X_" + str(n) + "_CqBoost.pck", 'rb')) for n in sizes]
    f1_scores_mean_CqBoost, f1_scores_var_CqBoost, accuracies_mean_CqBoost, accuracies_var_CqBoost = get_classifier_scores(X_CqBoost, y, sample_weights=class_weights)

    percent_same_features = calculer_similarites_features()

    print "Done tests CqBoost"

    with open(pickle_path + "f1_scores_mean_RFE.pck", 'wb') as f:
        pickle.dump(f1_scores_mean_RFE, f)

    with open(pickle_path + "f1_scores_mean_CqBoost.pck", 'wb') as f:
        pickle.dump(f1_scores_mean_CqBoost, f)

    with open(pickle_path + "f1_scores_var_RFE.pck", 'wb') as f:
        pickle.dump(f1_scores_var_RFE, f)

    with open(pickle_path + "f1_scores_var_CqBoost.pck", 'wb') as f:
        pickle.dump(f1_scores_var_CqBoost, f)

    with open(pickle_path + "accuracies_mean_RFE.pck", 'wb') as f:
        pickle.dump(accuracies_mean_RFE, f)

    with open(pickle_path + "accuracies_mean_CqBoost.pck", 'wb') as f:
        pickle.dump(accuracies_mean_CqBoost, f)

    with open(pickle_path + "accuracies_var_RFE.pck", 'wb') as f:
        pickle.dump(accuracies_var_RFE, f)

    with open(pickle_path + "accuracies_var_CqBoost.pck", 'wb') as f:
        pickle.dump(accuracies_var_CqBoost, f)

    with open(pickle_path + "percent_same_features.pck", 'wb') as f:
        pickle.dump(percent_same_features, f)

    print "Done tests comparison"


def determiner_poids_classes(y):
    n_neg = np.count_nonzero(y == -1)
    n_pos = np.count_nonzero(y == 1)
    n_samples = len(y)
    w_neg = n_samples/(2. * n_neg)
    w_pos = n_samples/(2. * n_pos)
    return np.asarray([w_neg if i == -1 else w_pos for i in y])


def load_data():
    X = np.load(open(pickle_path + 'X.pck', 'rb'))
    y = pickle.load(open(pickle_path + 'y.pck', 'rb'))
    return X, y


def choisir_features_CFS():
    X, y = load_data()
    print("Début Prudi Prudi")
    estimators_generator = CustomStumpsClassifiersGenerator()
    weight_vector = determiner_poids_classes(y)
    learner = CqBoostClassifier(estimators_generator=estimators_generator, classes_weights=weight_vector, n_max_iterations=4096)
    learner.fit(X, y, from_pickle=True)
    print("Fin Prudi Prudi")

    retained_features = np.asarray([e.attribute_index for e in learner.estimators_generator.estimators_])

    np.save(open(pickle_path + "retained_features_CqBoost.pck", 'wb'), retained_features)


def choisir_5000_features_RFE():
    X, y = load_data()
    print("Début RFE 5000")
    estimator = SVC(kernel='linear', class_weight='balanced')
    selector = RFE(estimator, n_features_to_select=5000, step=0.001)
    selector.fit(X, y)
    print("Fin RFE 5000")
    retained_scores_RFE = selector.ranking_
    retained_features_RFE_5000 = np.where(retained_scores_RFE == 1)[0]

    np.save(open(pickle_path + "retained_features_RFE_5000.pck", 'wb'), retained_features_RFE_5000)

    X_5000_RFE = X[:, retained_features_RFE_5000]

    np.save(open(pickle_path + "X_RFE_5000.pck", 'wb'), X_5000_RFE)

def choisir_derniers_features_RFE():
    _, y = load_data()
    X_5000_RFE = np.load(open(pickle_path + "X_RFE_5000.pck", 'rb'))
    print("Début RFE")
    estimator = SVC(kernel='linear', class_weight='balanced')
    selector = RFE(estimator, n_features_to_select=2, step=1)
    selector.fit(X_5000_RFE, y)
    print("Fin RFE")
    retained_scores_RFE = selector.ranking_

    np.save(open(pickle_path + "retained_scores_RFE.pck", 'wb'), retained_scores_RFE)


def choisir_features_RFE():
    choisir_5000_features_RFE()

    choisir_derniers_features_RFE()

def calculer_sizes():
    retained_features_CqBoost = np.load(open(pickle_path + "retained_features_CqBoost.pck", 'rb'))

    sizes = [2 ** i for i in range(10000) if 2 ** i < len(retained_features_CqBoost)]
    sizes.append(len(retained_features_CqBoost))

    return sizes


def generer_sous_ensembles_X_from_features_retained():
    retained_features_RFE = np.load(open(pickle_path + "retained_features_RFE.pck", 'rb'))

    retained_features_CqBoost = np.load(open(pickle_path + "retained_features_CqBoost.pck", 'rb'))

    X, _ = load_data()

    X_5000_RFE = np.load(open(pickle_path + "X_RFE_5000.pck", 'rb'))

    retained_features_RFE_5000 = np.load(open(pickle_path + "retained_features_RFE_5000.pck", 'rb'))

    sizes = calculer_sizes()

    retained_features_RFE_sizes, retained_features_CqBoost_sizes = [], []

    for s in sizes:
        retained_features_CqBoost_sizes.append(retained_features_CqBoost[:s])
        np.save(open(pickle_path + "X_" + str(s) + "_CqBoost.pck", 'wb'), X[:, retained_features_CqBoost[:s]])

    for s in sizes:
        retained_features_RFE_sizes.append(retained_features_RFE_5000[retained_features_RFE[:s]])
        np.save(open(pickle_path + "X_" + str(s) + "_RFE.pck", 'wb'), X_5000_RFE[:, retained_features_RFE[:s]])

    print "done sous-ensembles"


def see_tests_results():
    f1_scores_mean_RFE = {}
    with open(pickle_path + "f1_scores_mean_RFE.pck", 'rb') as f:
        f1_scores_mean_RFE = pickle.load(f)

    f1_scores_mean_CqBoost = {}
    with open(pickle_path + "f1_scores_mean_CqBoost.pck", 'rb') as f:
        f1_scores_mean_CqBoost = pickle.load(f)

    f1_scores_var_RFE = {}
    with open(pickle_path + "f1_scores_var_RFE.pck", 'rb') as f:
        f1_scores_var_RFE = pickle.load(f)

    f1_scores_var_CqBoost = {}
    with open(pickle_path + "f1_scores_var_CqBoost.pck", 'rb') as f:
        f1_scores_var_CqBoost = pickle.load(f)

    accuracies_mean_RFE = {}
    with open(pickle_path + "accuracies_mean_RFE.pck", 'rb') as f:
        accuracies_mean_RFE = pickle.load(f)

    accuracies_mean_CqBoost = {}
    with open(pickle_path + "accuracies_mean_CqBoost.pck", 'rb') as f:
        accuracies_mean_CqBoost = pickle.load(f)

    accuracies_var_RFE = {}
    with open(pickle_path + "accuracies_var_RFE.pck", 'rb') as f:
        accuracies_var_RFE = pickle.load(f)

    accuracies_var_CqBoost = {}
    with open(pickle_path + "accuracies_var_CqBoost.pck", 'rb') as f:
        accuracies_var_CqBoost = pickle.load(f)

    percent_same_features = {}
    with open(pickle_path + "percent_same_features.pck", 'rb') as f:
        percent_same_features = pickle.load(f)

    print "Mets un breakpoint ici si tu veux voir les dictionnaires mon chum"


def main_execution():
    # choisir_features_CFS()
    #
    # choisir_features_RFE()
    #
    # generer_sous_ensembles_X_from_features_retained()

    run_tests_comparison()

    # see_tests_results()

    print "Done"


if __name__ == '__main__':
    main_execution()

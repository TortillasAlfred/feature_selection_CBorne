# -*- coding: utf-8 -*-

from pyscm import SetCoveringMachineClassifier

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.utils import shuffle
import numpy as np

from CustomCqBoost import CqBoostClassifier

from StumpsGenerator import CustomStumpsClassifiersGenerator

import argparse

import pickle
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


def get_random_states():
    return [1, 7, 18, 22, 33, 42, 69, 100, 420, 666]

def wa(y_true, y_pred, weights):
    w_y = np.asarray(y_true, dtype='float64')
    for k in weights.keys():
        w_y[y_true == k] = k * weights[k]
    correct = np.multiply(y_pred, w_y)
    return correct[correct > 0].sum()/correct.shape[0]


def get_classifier_scores(X_train, X_test, y_train, y_test):
    f1_scores = {}
    accuracies = {}
    # weight_dict = determiner_poids_classes(y_train, return_dict=True)

    for algo in algos_tested:
        print("Begin " + str(algo))

        for i in range(len(X_train)):
            x_train = X_train[i].copy()
            x_test = X_test[i].copy()
            learner = algos_tested[algo]
            cv_params = params[algo]
            # scorer = make_scorer(wa, weights=weight_dict)
            # Cross validation avec weighted accuracy
            clf = GridSearchCV(learner, cv_params, verbose=1, n_jobs=-1, cv=5)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            f1_score_algo = f1_score(y_test, y_pred)
            # accuracies_temp.append(wa(y_test, y_pred, weights=weight_dict))
            accuracy_algo = accuracy_score(y_test, y_pred)

        f1_scores[algo] = f1_score_algo
        accuracies[algo] = accuracy_algo

    return f1_scores, accuracies


def calculer_similarites_features():
    retained_features_CqBoost = np.load(pickle_path + "retained_features_CqBoost.npy")
    retained_features_RFE = np.load(pickle_path + "retained_features_RFE.npy")
    retained_features_RFE_5000 = np.load(pickle_path + "retained_features_RFE_5000.npy")

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

def make_save_split(X, y, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    np.save(pickle_path + "X_train", X_train)
    np.save(pickle_path + "X_test", X_test)
    np.save(pickle_path + "y_train", y_train)
    np.save(pickle_path + "y_test", y_test)


def run_tests_random_state(n):
    global pickle_path
    X = np.load(pickle_path + "X_balanced.npy")
    y = np.load(pickle_path + "y_balanced.npy")

    rs = get_random_states()[n]

    pickle_path += str(rs) + '/'

    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)

    make_save_split(X, y, rs)

    choisir_features_CFS()

    choisir_features_RFE()

    generer_sous_ensembles_X_from_features_retained()

    run_tests_comparison()


def run_tests_comparison():
    _, y_train, _, y_test = load_data()
    # class_weights = determiner_poids_classes(y)

    sizes = calculer_sizes()

    X_RFE_train = [np.load(pickle_path + "X_" + str(n) + "_train_RFE.npy") for n in sizes]
    X_RFE_test = [np.load(pickle_path + "X_" + str(n) + "_test_RFE.npy") for n in sizes]

    covariance_RFE = np.asarray([np.tril(np.corrcoef(x, rowvar=False), k=-1) for x in X_RFE_train[1:]])

    covariance_RFE_abs = np.absolute(covariance_RFE)

    cov_RFE = np.asarray([np.sum(cov)/(len(cov) * (len(cov) - 1)) * 2. for cov in covariance_RFE])

    cov_RFE_abs = np.asarray([np.sum(cov)/(len(cov) * (len(cov) - 1)) * 2. for cov in covariance_RFE_abs])

    f1_scores_mean_RFE, accuracies_mean_RFE= get_classifier_scores(X_RFE_train, X_RFE_test, y_train, y_test)

    print("Done tests RFE")

    X_CqBoost_train = [np.load(pickle_path + "X_" + str(n) + "_train_CqBoost.npy") for n in sizes]
    X_CqBoost_test = [np.load(pickle_path + "X_" + str(n) + "_test_CqBoost.npy") for n in sizes]

    covariance_CqBoost = np.asarray([np.tril(np.corrcoef(x, rowvar=False), k=-1) for x in X_CqBoost_train[1:]])

    covariance_CqBoost_abs = np.absolute(covariance_CqBoost)

    cov_CqBoost = np.asarray([np.sum(cov) / (len(cov) * (len(cov) - 1)) * 2. for cov in covariance_CqBoost])

    cov_CqBoost_abs = np.asarray([np.sum(cov) / (len(cov) * (len(cov) - 1)) * 2. for cov in covariance_CqBoost_abs])

    f1_scores_mean_CqBoost, accuracies_mean_CqBoost = get_classifier_scores(X_CqBoost_train, X_CqBoost_test, y_train, y_test)

    percent_same_features = calculer_similarites_features()

    print("Done tests CqBoost")

    with open(pickle_path + "f1_scores_mean_RFE.pck", 'wb') as f:
        pickle.dump(f1_scores_mean_RFE, f)

    with open(pickle_path + "f1_scores_mean_CqBoost.pck", 'wb') as f:
        pickle.dump(f1_scores_mean_CqBoost, f)

    with open(pickle_path + "accuracies_mean_RFE.pck", 'wb') as f:
        pickle.dump(accuracies_mean_RFE, f)

    with open(pickle_path + "accuracies_mean_CqBoost.pck", 'wb') as f:
        pickle.dump(accuracies_mean_CqBoost, f)

    with open(pickle_path + "percent_same_features.pck", 'wb') as f:
        pickle.dump(percent_same_features, f)

    np.save(pickle_path + "cov_RFE", cov_RFE)
    np.save(pickle_path + "cov_RFE_abs", cov_RFE_abs)

    np.save(pickle_path + "cov_CqBoost", cov_CqBoost)
    np.save(pickle_path + "cov_CqBoost_abs", cov_CqBoost_abs)

    np.save(pickle_path + "sizes", sizes)

    print("Done tests comparison")


def determiner_poids_classes(y, return_dict=False):
    n_neg = np.count_nonzero(y == -1)
    n_pos = np.count_nonzero(y == 1)
    n_samples = len(y)
    w_neg = n_samples/(2. * n_neg)
    w_pos = n_samples/(2. * n_pos)
    if return_dict:
        return {-1: w_neg, 1: w_pos}
    else:
        return np.asarray([w_neg if i == -1 else w_pos for i in y])


def load_data():
    X_train = np.load(pickle_path + 'X_train.npy')
    X_test = np.load(pickle_path + 'X_test.npy')
    y_train = np.load(pickle_path + 'y_train.npy')
    y_test = np.load(pickle_path + 'y_test.npy')
    return X_train, y_train, X_test, y_test


def choisir_features_CFS():
    X, y, _, _ = load_data()
    print("Début Prudi Prudi")
    estimators_generator = CustomStumpsClassifiersGenerator(pickle_path)
    # weight_vector = determiner_poids_classes(y)
    learner = CqBoostClassifier(estimators_generator=estimators_generator, n_max_iterations=4096)
    learner.fit(X, y, from_pickle=False)
    print("Fin Prudi Prudi")

    retained_features = np.asarray([e.attribute_index for e in learner.estimators_generator.estimators_])

    np.save(pickle_path + "retained_features_CqBoost", retained_features)


def choisir_5000_features_RFE():
    X_train, y_train, X_test, _ = load_data()
    print("Début RFE 5000")
    estimator = SVC(kernel='linear', class_weight='balanced')
    selector = RFE(estimator, n_features_to_select=5000, step=0.01, verbose=1)
    selector.fit(X_train, y_train)
    print("Fin RFE 5000")
    retained_scores_RFE = selector.ranking_
    retained_features_RFE_5000 = np.where(retained_scores_RFE == 1)[0]

    np.save(pickle_path + "retained_features_RFE_5000", retained_features_RFE_5000)

    X_5000_train_RFE = X_train[:, retained_features_RFE_5000]
    np.save(pickle_path + "X_train_RFE_5000", X_5000_train_RFE)
    X_5000_test_RFE = X_test[:, retained_features_RFE_5000]
    np.save(pickle_path + "X_test_RFE_5000", X_5000_test_RFE)


def choisir_derniers_features_RFE():
    _, y_train, X_test, _ = load_data()
    X_5000_train_RFE = np.load(pickle_path + "X_train_RFE_5000.npy")
    print("Début RFE")
    estimator = SVC(kernel='linear', class_weight='balanced')
    selector = RFE(estimator, n_features_to_select=2, step=1)
    selector.fit(X_5000_train_RFE, y_train)
    print("Fin RFE")
    retained_scores_RFE = selector.ranking_

    np.save(pickle_path + "retained_scores_RFE", retained_scores_RFE)

    retained_features_RFE = np.where(retained_scores_RFE == 1)[0]
    for i in range(2, 5000):
        retained_features_RFE = np.append(retained_features_RFE, np.where(retained_scores_RFE == i)[0])

    np.save(pickle_path + "retained_features_RFE", retained_features_RFE)


def choisir_features_RFE():
    choisir_5000_features_RFE()

    choisir_derniers_features_RFE()


def calculer_sizes():
    retained_features_CqBoost = np.load(pickle_path + "retained_features_CqBoost.npy")

    sizes = [2 ** i for i in range(10000) if 2 ** i < len(retained_features_CqBoost)]

    return sizes


def generer_sous_ensembles_X_from_features_retained():
    retained_features_RFE = np.load(pickle_path + "retained_features_RFE.npy")

    retained_features_CqBoost = np.load(pickle_path + "retained_features_CqBoost.npy")

    X_train, _, X_test, _ = load_data()

    X_5000_train_RFE = np.load(pickle_path + "X_train_RFE_5000.npy")
    X_5000_test_RFE = np.load(pickle_path + "X_test_RFE_5000.npy")

    retained_features_RFE_5000 = np.load(pickle_path + "retained_features_RFE_5000.npy")

    sizes = calculer_sizes()

    retained_features_RFE_sizes, retained_features_CqBoost_sizes = [], []

    for s in sizes:
        retained_features_CqBoost_sizes.append(retained_features_CqBoost[:s])
        np.save(pickle_path + "X_" + str(s) + "_train_CqBoost", X_train[:, retained_features_CqBoost[:s]])
        np.save(pickle_path + "X_" + str(s) + "_test_CqBoost", X_test[:, retained_features_CqBoost[:s]])

    for s in sizes:
        retained_features_RFE_sizes.append(retained_features_RFE_5000[retained_features_RFE[:s]])
        np.save(pickle_path + "X_" + str(s) + "_train_RFE", X_5000_train_RFE[:, retained_features_RFE[:s]])
        np.save(pickle_path + "X_" + str(s) + "_test_RFE", X_5000_test_RFE[:, retained_features_RFE[:s]])

    print("done sous-ensembles")


def make_undersampled_balanced_dataset(random_state):
    X, y = load_data()

    X_pos = X[y == 1]
    X_neg = X[y == -1]
    y_pos = y[y == 1]
    y_neg = y[y == -1]

    np.random.seed(random_state)

    n_pos = X_pos.shape[0]
    ind = np.arange(X_neg.shape[0])
    np.random.shuffle(ind)
    X_neg_retenus = X_neg[ind[:n_pos], :]
    y_neg_retenus = y_neg[ind[:n_pos]]

    X_balanced = np.concatenate((X_pos, X_neg_retenus), axis=0)
    y_balanced = np.concatenate((y_pos, y_neg_retenus), axis=0)

    X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=random_state)

    np.save(pickle_path + "X_balanced", X_balanced)
    np.save(pickle_path + "y_balanced", y_balanced)


def main_execution(n):
    run_tests_random_state(n)

    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=0, type=int, nargs='?', help='Index of the desired rs in the rs list.')
    parser.add_argument('--rs', default=[42], type=int, nargs='?', help='The random states list (rs)')
    parser.add_argument('--dataset', default='movielens', type=str, help='The name of the dataset for the experiments')
    parser.add_argument('--outdir', default='results/test', type=str,
                        help='The name of the output directory for the experiments')
    parser.add_argument('--algos',
                        default=['hyperkrr'],  # ['fskrr', 'metakrr', 'multitask', 'snail', 'mann', 'maml'],
                        type=str, nargs='+',
                        help='The name of the algos: fskrr|metakrr|multitask|snail|mann|maml')
    args = parser.parse_args()
    main_execution(args.n)
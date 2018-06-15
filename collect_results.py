import os

from tests import pickle_path, get_random_states, algos_tested

import numpy as np
import pickle
import csv

desired_outputs_pck = ['accuracies_mean_CqBoost', 'accuracies_mean_RFE',
                       'f1_scores_mean_CqBoost', 'f1_scores_mean_RFE']

desired_outputs_npy = ['cov_CqBoost', 'cov_RFE', 'percent_same_features']

def main_execution():
    global pickle_path

    experiments_npy = []
    experiments_pck = []

    for rs in get_random_states():
        experiment_npy = {}
        experiment_pck = {}
        current_dir = pickle_path + str(rs) + '/'
        for file in os.listdir(current_dir):
            filename = os.path.splitext(file)[0]
            if filename in desired_outputs_npy:
                npy_array = np.load(current_dir + file)
                experiment_npy[filename] = npy_array

            if filename in desired_outputs_pck:
                pck_dict = pickle.load(open(current_dir + file, 'rb'))
                experiment_pck[filename] = pck_dict

        experiments_npy.append(experiment_npy)
        experiments_pck.append(experiment_pck)

    # Determiner le nombre max de rangees
    n_inputs = min([len(dict['percent_same_features']) for dict in experiments_npy])

    metriques_dict = {}
    tests_dict = {}
    tests_dict["percent_same_features"] = np.mean([dict['percent_same_features'][:n_inputs] for dict in experiments_npy], axis=0)
    tests_dict["cov_CqBoost"] = np.mean([dict['cov_CqBoost'][:n_inputs-1] for dict in experiments_npy], axis=0)
    tests_dict["cov_RFE"] = np.mean([dict['cov_RFE'][:n_inputs-1] for dict in experiments_npy], axis=0)

    for algo in algos_tested:
        metriques_algo = {}

        metriques_algo["accuracies_mean_CqBoost"] = np.mean([dict['accuracies_mean_CqBoost'][algo][:n_inputs] for dict in experiments_pck], axis=0)
        metriques_algo["accuracies_var_CqBoost"] = np.var([dict['accuracies_mean_CqBoost'][algo][:n_inputs] for dict in experiments_pck], axis=0)
        metriques_algo["accuracies_mean_RFE"] = np.mean([dict['accuracies_mean_RFE'][algo][:n_inputs] for dict in experiments_pck], axis=0)
        metriques_algo["accuracies_var_RFE"] = np.var([dict['accuracies_mean_RFE'][algo][:n_inputs] for dict in experiments_pck], axis=0)
        metriques_algo["f1_scores_mean_CqBoost"] = np.mean([dict['f1_scores_mean_CqBoost'][algo][:n_inputs] for dict in experiments_pck], axis=0)
        metriques_algo["f1_scores_var_CqBoost"] = np.var([dict['f1_scores_mean_CqBoost'][algo][:n_inputs] for dict in experiments_pck], axis=0)
        metriques_algo["f1_scores_mean_RFE"] = np.mean([dict['f1_scores_mean_RFE'][algo][:n_inputs] for dict in experiments_pck], axis=0)
        metriques_algo["f1_scores_var_RFE"] = np.var([dict['f1_scores_mean_RFE'][algo][:n_inputs] for dict in experiments_pck], axis=0)

        metriques_dict[algo] = metriques_algo

    with open(pickle_path + 'test_results_.csv', 'w') as f:
        writer = csv.writer(f)
        for k, value in tests_dict.items():
            writer.writerow([k])
            for v in value:
                writer.writerow([v])

        for algo in algos_tested:
            writer.writerow([algo])
            dict_algo = metriques_dict[algo]
            writer.writerow([k for k in dict_algo.keys()])
            vals = [d for d in dict_algo.values()]
            for i in range(len(vals[0])):
                writer.writerow([v[i] for v in vals])


if __name__ == '__main__':
    main_execution()
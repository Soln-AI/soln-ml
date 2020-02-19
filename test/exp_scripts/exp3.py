import os
import sys
import time
import pickle
import argparse
import tabulate
import shutil
import numpy as np
import autosklearn.classification
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())

from automlToolkit.bandits.first_layer_bandit import FirstLayerBandit
from automlToolkit.datasets.utils import load_train_test_data
from automlToolkit.components.utils.constants import CATEGORICAL
from automlToolkit.components.ensemble.ensemble_builder import EnsembleBuilder

parser = argparse.ArgumentParser()
dataset_set = 'yeast,vehicle,diabetes,spectf,credit,' \
              'ionosphere,lymphography,messidor_features,winequality_red,fri_c1'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--methods', type=str, default='hmab')
parser.add_argument('--algo_num', type=int, default=15)
parser.add_argument('--trial_num', type=int, default=100)
parser.add_argument('--rep_num', type=int, default=5)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--time_costs', type=str, default='1200')
parser.add_argument('--ensemble', type=int, choices=[0, 1], default=0)
parser.add_argument('--eval_type', type=str, choices=['cv', 'holdout'], default='cv')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--mode', type=str, default='both', choices=['fe', 'hpo', 'both'])

save_dir = './data/exp_results/exp3/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

per_run_time_limit = 1200


def evaluate_hmab(algorithms, run_id, dataset='credit', trial_num=200, seed=1, eval_type='holdout', mode='both'):
    task_id = '%s-hmab-%d-%d-%s' % (dataset, len(algorithms), trial_num, mode)
    _start_time = time.time()
    raw_data, test_raw_data = load_train_test_data(dataset)
    bandit = FirstLayerBandit(trial_num, algorithms, raw_data,
                              output_dir='logs/%s/' % task_id,
                              per_run_time_limit=per_run_time_limit,
                              dataset_name='%s-%d' % (dataset, run_id),
                              seed=seed,
                              eval_type=eval_type,
                              mode=mode)
    bandit.optimize()
    time_cost = int(time.time() - _start_time)
    print(bandit.final_rewards)
    print(bandit.action_sequence)

    validation_accuracy = np.max(bandit.final_rewards)
    test_accuracy = bandit.score(test_raw_data)
    test_accuracy_with_ens = EnsembleBuilder(bandit).score(test_raw_data)

    print('Dataset          : %s' % dataset)
    print('Validation/Test score : %f - %f' % (validation_accuracy, test_accuracy))
    print('Test score with ensem : %f' % test_accuracy_with_ens)

    save_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
    with open(save_path, 'wb') as f:
        stats = [time_cost, test_accuracy_with_ens, bandit.time_records, bandit.final_rewards]
        pickle.dump([validation_accuracy, test_accuracy, stats], f)
    return time_cost


def load_hmab_time_costs(start_id, rep, dataset, n_algo, trial_num):
    task_id = '%s-hmab-%d-%d' % (dataset, n_algo, trial_num)
    time_costs = list()
    for run_id in range(start_id, start_id + rep):
        save_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
        with open(save_path, 'rb') as f:
            time_cost = pickle.load(f)[2][0]
            time_costs.append(time_cost)
    assert len(time_costs) == rep
    return time_costs


def check_datasets(datasets):
    for _dataset in datasets:
        try:
            _, _ = load_train_test_data(_dataset, random_state=1)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    algo_num = args.algo_num
    trial_num = args.trial_num
    start_id = args.start_id
    rep = args.rep_num
    mode = args.mode
    methods = args.methods.split(',')
    time_costs = [int(item) for item in args.time_costs.split(',')]
    eval_type = args.eval_type
    enable_ensemble = bool(args.ensemble)

    # Prepare random seeds.
    np.random.seed(args.seed)
    seeds = np.random.randint(low=1, high=10000, size=start_id + args.rep_num)

    if algo_num == 4:
        algorithms = ['k_nearest_neighbors', 'libsvm_svc', 'random_forest', 'adaboost']
    elif algo_num == 8:
        algorithms = ['passive_aggressive', 'k_nearest_neighbors', 'libsvm_svc', 'sgd',
                      'adaboost', 'random_forest', 'extra_trees', 'decision_tree']
    elif algo_num == 15:
        algorithms = ['adaboost', 'random_forest',
                      'libsvm_svc', 'sgd',
                      'extra_trees', 'decision_tree',
                      'liblinear_svc', 'k_nearest_neighbors',
                      'passive_aggressive', 'xgradient_boosting',
                      'lda', 'qda',
                      'multinomial_nb', 'gaussian_nb', 'bernoulli_nb'
                      ]
    else:
        raise ValueError('Invalid algorithm num - %d!' % algo_num)

    dataset_list = dataset_str.split(',')
    check_datasets(dataset_list)

    for dataset in dataset_list:
        for mth in methods:
            if mth == 'plot':
                break

            for run_id in range(start_id, start_id + rep):
                seed = int(seeds[run_id])
                if mth == 'hmab':
                    time_cost = evaluate_hmab(algorithms, run_id, dataset,
                                              trial_num=trial_num, seed=seed,
                                              eval_type=eval_type,

                                              )
                else:
                    raise ValueError('Invalid method name: %s.' % mth)

    if methods[-1] == 'plot':
        headers = ['dataset']
        ausk_id = 'ausk-ens%d' % enable_ensemble
        method_ids = ['hmab']
        for mth in method_ids:
            headers.extend(['val-%s' % mth, 'test-%s' % mth])

        tbl_data = list()
        for dataset in dataset_list:
            row_data = [dataset]
            for mth in method_ids:
                results = list()
                for run_id in range(rep):
                    task_id = '%s-%s-%d-%d-%s' % (dataset, mth, len(algorithms), trial_num, mode)
                    file_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
                    if not os.path.exists(file_path):
                        continue
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    val_acc, test_acc, _tmp = data
                    if enable_ensemble and mth == 'hmab':
                        test_acc = _tmp[1]
                    results.append([val_acc, test_acc])
                if len(results) == rep:
                    results = np.array(results)
                    stats_ = zip(np.mean(results, axis=0), np.std(results, axis=0))
                    string = ''
                    for mean_t, std_t in stats_:
                        string += u'%.3f\u00B1%.3f |' % (mean_t, std_t)
                    print(dataset, mth, '=' * 30)
                    print('%s-%s: mean\u00B1std' % (dataset, mth), string)
                    print('%s-%s: median' % (dataset, mth), np.median(results, axis=0))

                    for idx in range(results.shape[1]):
                        vals = results[:, idx]
                        median = np.median(vals)
                        if median == 0.:
                            row_data.append('-')
                        else:
                            row_data.append(u'%.4f' % median)
                else:
                    row_data.extend(['-'] * 2)

            tbl_data.append(row_data)
        print(tabulate.tabulate(tbl_data, headers, tablefmt='github'))

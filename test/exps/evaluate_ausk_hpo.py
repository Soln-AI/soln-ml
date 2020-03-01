import os
import numpy as np
import sys
import argparse
import pickle
import time

from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from sklearn.metrics import accuracy_score
from autosklearn.classification import AutoSklearnClassifier

sys.path.append(os.getcwd())
from automlToolkit.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from automlToolkit.datasets.utils import load_train_test_data
from automlToolkit.components.evaluators.evaluator import Evaluator, fetch_predict_estimator

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='credit')
parser.add_argument('--algo', type=str, default='random_forest')
parser.add_argument('--iter_num', type=int, default=100)
parser.add_argument('--rep_num', type=int, default=5)
parser.add_argument('--seed', type=int, default=1)

save_dir = './data/exp_results/overfit/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def conduct_hpo(dataset='pc4', classifier_id='random_forest', iter_num=100, run_id=0, seed=1):
    from autosklearn.pipeline.components.classification import _classifiers
    task_id = 'hpo-%s-%s-%d' % (dataset, classifier_id, iter_num)
    save_path = save_dir + '%s-%d.pkl' % (task_id, run_id)

    clf_class = _classifiers[classifier_id]
    cs = clf_class.get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", classifier_id)
    cs.add_hyperparameter(model)

    raw_data, test_raw_data = load_train_test_data(dataset, random_state=seed)
    evaluator = Evaluator(cs.get_default_configuration(), name='hpo', data_node=raw_data,
                          resampling_strategy='cv', seed=seed)

    optimizer = SMACOptimizer(
        evaluator, cs, trials_per_iter=2,
        output_dir='logs', per_run_time_limit=180,
        evaluation_limit=iter_num * 2
    )

    start_time = time.time()
    config, perf = optimizer.optimize()
    time_cost = time.time() - start_time

    estimator = fetch_predict_estimator(config, raw_data.data[0], raw_data.data[1])
    pred = estimator.predict(test_raw_data.data[0])
    test_perf = accuracy_score(test_raw_data.data[1], pred)
    print(perf)
    print(test_perf)

    with open(save_path, 'wb') as f:
        pickle.dump([perf, test_perf], f)

    print("hmab %f seconds" % time_cost)
    return time_cost


def conduct_ausk(dataset='pc4', classifier_id='random_forest', iter_num=100, run_id=0, seed=1, time_limit=3600):
    task_id = 'hpo-%s-%s-%d' % (dataset, classifier_id, iter_num)
    save_path = save_dir + '%ausk-s-%d.pkl' % (task_id, run_id)

    automl = AutoSklearnClassifier(
        time_left_for_this_task=int(time_limit),
        include_preprocessors=['no_preprocessing'],
        n_jobs=1,
        include_estimators=['random_forest'],
        ensemble_memory_limit=8192,
        ml_memory_limit=8192,
        ensemble_size=1,
        ensemble_nbest=1,
        initial_configurations_via_metalearning=0,
        per_run_time_limit=30,
        seed=seed,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5}
    )

    raw_data, test_raw_data = load_train_test_data(dataset, random_state=seed)
    X, y = raw_data.data
    automl.fit(X.copy(), y.copy())
    best_result = np.max(automl.cv_results_['mean_test_score'])
    print(best_result)
    X_test, y_test = test_raw_data.data
    pred = automl.predict(X_test)
    test_perf = accuracy_score(y_test, pred)

    with open(save_path, 'wb') as f:
        pickle.dump([best_result, test_perf], f)


def check_datasets(datasets):
    for _dataset in datasets:
        try:
            _, _ = load_train_test_data(_dataset, random_state=1)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


if __name__ == '__main__':
    args = parser.parse_args()
    datasets = args.datasets
    iter_num = args.iter_num
    rep = args.rep_num
    algo = args.algo

    dataset_list = datasets.split(',')
    check_datasets(dataset_list)

    mode_list = ['hmab', 'ausk']
    for dataset in dataset_list:
        for run_id in range(rep):
            time_limit = None
            for mode in mode_list:
                if mode == 'hmab':
                    time_limit = conduct_hpo(dataset=dataset, classifier_id=algo, iter_num=iter_num, run_id=run_id)
                elif mode == 'ausk':
                    conduct_ausk(dataset=dataset, classifier_id=algo, iter_num=iter_num, run_id=run_id,
                                 time_limit=time_limit)

import os
import sys
import time
import pickle
import argparse
import tabulate
import autosklearn.classification
from sklearn.metrics import accuracy_score as acc

sys.path.append(os.getcwd())
from automlToolkit.estimators import Classifier
from automlToolkit.datasets.utils import load_train_test_data

parser = argparse.ArgumentParser()
dataset_set = 'yeast,vehicle,diabetes,spectf,credit,' \
              'ionosphere,lymphography,messidor_features,winequality_red,fri_c1'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--methods', type=str, default='hmab,ausk')
parser.add_argument('--ens', type=str, default="None,bagging,ensemble_selection,stacking")
parser.add_argument('--rep_num', type=int, default=5)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--time_costs', type=int, default='600')

save_dir = './data/exp_results/exp1/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

per_run_time_limit = 300


def evaluate_hmab(algorithms, run_id, dataset='credit', time_limit=600, ens_method=None, ens_size=20):
    algo = algorithms.copy()
    algo.append('lightgbm')
    task_id = '%s-hmab-%d-%d-%s' % (dataset, len(algo), time_limit, str(ens_method))
    _start_time = time.time()
    raw_data, test_raw_data = load_train_test_data(dataset)
    clf = Classifier(metric='acc', time_limit=time_limit,
                     iter_num_per_algo=100, include_algorithms=algo,
                     ensemble_method=ens_method, ensemble_size=ens_size,
                     per_run_time_limit=per_run_time_limit, random_state=run_id)
    clf.fit(raw_data)
    time_cost = int(time.time() - _start_time)
    print(clf._ml_engine.best_perf)

    validation_accuracy = clf._ml_engine.best_perf
    pred = clf.predict(test_raw_data)
    test_accuracy = acc(pred, test_raw_data.data[1])

    print('Dataset          : %s' % dataset)
    print('Validation/Test score : %f - %f' % (validation_accuracy, test_accuracy))

    save_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
    with open(save_path, 'wb') as f:
        pickle.dump([validation_accuracy, test_accuracy, time_cost], f)
    return time_cost


def load_hmab_time_costs(start_id, rep, dataset, n_algo, time_limit):
    task_id = '%s-hmab-%d-%d-None' % (dataset, n_algo, time_limit)
    time_costs = list()
    for run_id in range(start_id, start_id + rep):
        save_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
        with open(save_path, 'rb') as f:
            time_cost = pickle.load(f)[2]
            time_costs.append(time_cost)
    assert len(time_costs) == rep
    return time_costs


def evaluate_autosklearn(algorithms, rep_id,
                         dataset='credit', time_limit=600,
                         enable_ens=True, enable_meta_learning=False):
    algo = algorithms.copy()
    algo.append('LightGBM')
    print('%s\nDataset: %s, Run_id: %d, Budget: %d.\n%s' % ('=' * 50, dataset, rep_id, time_limit, '=' * 50))
    task_id = '%s-ausk-%d-%d' % (dataset, len(algo), int(time_limit))
    if enable_ens:
        ensemble_size, ensemble_nbest = 20, 5
    else:
        ensemble_size, ensemble_nbest = 1, 1
    if enable_meta_learning:
        init_config_via_metalearning = 25
    else:
        init_config_via_metalearning = 0

    include_models = algo

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=int(time_limit),
        per_run_time_limit=per_run_time_limit,
        n_jobs=1,
        include_estimators=include_models,
        ensemble_memory_limit=16384,
        ml_memory_limit=16384,
        ensemble_size=ensemble_size,
        ensemble_nbest=ensemble_nbest,
        initial_configurations_via_metalearning=init_config_via_metalearning,
        seed=int(rep_id),
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67})

    print(automl)
    raw_data, test_raw_data = load_train_test_data(dataset)
    X, y = raw_data.data
    X_test, y_test = test_raw_data.data
    feat_type = ['Categorical' if _type == CATEGORICAL else 'Numerical'
                 for _type in raw_data.feature_types]
    automl.fit(X.copy(), y.copy(), feat_type=feat_type)
    model_desc = automl.show_models()
    str_stats = automl.sprint_statistics()
    valid_results = automl.cv_results_['mean_test_score']
    validation_accuracy = np.max(valid_results)

    # Test performance.
    automl.refit(X.copy(), y.copy())
    predictions = automl.predict(X_test)
    test_accuracy = acc(y_test, predictions)

    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(str_stats)
    print(model_desc)
    print('Validation Accuracy:', validation_accuracy)
    print("Test Accuracy      :", test_accuracy)

    save_path = save_dir + '%s-%d.pkl' % (task_id, rep_id)
    with open(save_path, 'wb') as f:
        pickle.dump([validation_accuracy, test_accuracy, time_limit], f)


def check_datasets(datasets):
    for _dataset in datasets:
        try:
            _, _ = load_train_test_data(_dataset, random_state=1)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter
import numpy as np
from lightgbm import LGBMClassifier
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm

from automlToolkit.components.utils.constants import *


class LightGBM(AutoSklearnClassificationAlgorithm):
    def __init__(self, n_estimators, learning_rate, num_leaves, min_child_weight,
                 subsample, colsample_bytree, reg_alpha, reg_lambda, random_state):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree

        self.n_jobs = -1
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        self.estimator = LGBMClassifier(num_leaves=self.num_leaves,
                                        learning_rate=self.learning_rate,
                                        n_estimators=self.n_estimators,
                                        min_child_weight=self.min_child_weight,
                                        subsample=self.subsample,
                                        colsample_bytree=self.colsample_bytree,
                                        reg_alpha=self.reg_alpha,
                                        reg_lambda=self.reg_lambda,
                                        random_state=self.random_state,
                                        n_jobs=self.n_jobs)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LightGBM Classifier',
                'name': 'LightGBM Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': False,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, default_value=500)
            num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 1023, default_value=31)
            learning_rate = UniformFloatHyperparameter("learning_rate", 0.025, 0.3, default_value=0.1, log=True)
            min_child_weight = UniformIntegerHyperparameter("min_child_weight", 1, 10, default_value=1)
            subsample = UniformFloatHyperparameter("subsample", 0.5, 1, default_value=1)
            colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.5, 1, default_value=1)
            reg_alpha = UniformFloatHyperparameter('reg_alpha', 1e-10, 10, log=True, default_value=1e-10)
            reg_lambda = UniformFloatHyperparameter("reg_lambda", 1e-10, 10, log=True, default_value=1e-10)
            cs.add_hyperparameters([n_estimators, num_leaves, learning_rate, min_child_weight, subsample,
                                    colsample_bytree, reg_alpha, reg_lambda])
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'n_estimators': hp.randint('lgb_n_estimators', 901) + 100,
                     'num_leaves': hp.randint('lgb_num_leaves', 993) + 31,
                     'learning_rate': hp.loguniform('lgb_learning_rate', np.log(0.025), np.log(0.3)),
                     'min_child_weight': hp.randint('lgb_min_child_weight', 10) + 1,
                     'subsample': hp.uniform('lgb_subsample', 0.5, 1),
                     'colsample_bytree': hp.uniform('lgb_colsample_bytree', 0.5, 1),
                     'reg_alpha': hp.loguniform('lgb_reg_alpha', np.log(1e-10), np.log(10)),
                     'reg_lambda': hp.loguniform('lgb_reg_lambda', np.log(1e-10), np.log(10))
                     }

            init_trial = {'n_estimators': 500,
                          'num_leaves': 31,
                          'learning_rate': 0.1,
                          'min_child_weight': 1,
                          'subsample': 1,
                          'colsample_bytree': 1,
                          'reg_alpha': 1e-10,
                          'reg_lambda': 1e-10
                          }

            return space


if __name__ == "__main__":
    autosklearn.pipeline.components.classification.add_classifier(LightGBM)
    args = parser.parse_args()
    dataset_str = args.datasets
    start_id = args.start_id
    rep = args.rep_num
    methods = args.methods.split(',')
    ens_list = args.ens.split(',')
    time_limit = args.time_costs

    # Prepare random seeds.
    np.random.seed(1)
    seeds = np.random.randint(low=1, high=10000, size=start_id + args.rep_num)

    algorithms = ['liblinear_svc', 'random_forest', 'k_nearest_neighbors', 'libsvm_svc']

    dataset_list = dataset_str.split(',')
    check_datasets(dataset_list)

    for dataset in dataset_list:
        for mth in methods:
            if mth == 'plot':
                break

            if mth.startswith('ausk'):
                time_costs = load_hmab_time_costs(start_id, rep, dataset, len(algorithms) + 1, time_limit)
                print(time_costs)
                median = np.median(time_costs)
                time_costs = [median] * rep
                print(median, time_costs)

            for run_id in range(start_id, start_id + rep):
                seed = int(seeds[run_id])
                if mth == 'hmab':
                    for ens in ens_list:
                        if ens == 'None':
                            ens = None
                        if ens in ['bagging', 'stacking']:
                            ens_size = 6
                        else:
                            ens_size = 20
                        time_cost = evaluate_hmab(algorithms, run_id, dataset, time_limit=time_limit, ens_method=ens,
                                                  ens_size=ens_size)
                elif mth == 'ausk':
                    time_cost = time_costs[run_id - start_id]
                    evaluate_autosklearn(algorithms, run_id, dataset, time_cost)
                else:
                    raise ValueError('Invalid method name: %s.' % mth)

    if methods[-1] == 'plot':
        headers = ['dataset']
        ausk_id = 'ausk'
        method_ids = ['hmab', ausk_id]
        for mth in method_ids:
            headers.extend(['val-%s' % mth, 'test-%s' % mth])

        tbl_data = list()
        for dataset in dataset_list:
            row_data = [dataset]
            for mth in method_ids:
                results = list()
                for ens in ens_list:
                    for run_id in range(rep):
                        task_id = '%s-%s-%d-%d-%s' % (dataset, mth, len(algorithms), time_limit, str(ens))
                        file_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
                        if not os.path.exists(file_path):
                            continue
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        val_acc, test_acc, _tmp = data
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

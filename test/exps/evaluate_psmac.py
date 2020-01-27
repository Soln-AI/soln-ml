import os
import sys
import argparse
import time
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter

sys.path.append(os.getcwd())
from automlToolkit.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from automlToolkit.components.hpo_optimizer.psmac_optimizer import PSMACOptimizer
from automlToolkit.datasets.utils import load_data
from automlToolkit.components.evaluator import Evaluator

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'
parser.add_argument('--dataset', type=str, default='diabetes')
parser.add_argument('--optimizer', type=str, default='smac', choices=['smac', 'psmac'])
parser.add_argument('--n', type=int, default=4)
parser.add_argument('--algo', type=str, default='extra_trees')
parser.add_argument('--runcount_limit', type=int, default=50)
parser.add_argument('--trial', type=int, default=8)
parser.add_argument('--rep_num', type=int, default=1)

parser.add_argument('--seed', type=int, default=1)


def conduct_hpo(optimizer='smac', dataset='pc4', classifier_id='random_forest', runcount_limit=100):
    from autosklearn.pipeline.components.classification import _classifiers

    clf_class = _classifiers[classifier_id]
    cs = clf_class.get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", classifier_id)
    cs.add_hyperparameter(model)

    raw_data = load_data(dataset, datanode_returned=True)
    print(set(raw_data.data[1]))
    evaluator = Evaluator(cs.get_default_configuration(), name='hpo', data_node=raw_data)

    perf = 0
    start_time = time.time()
    for i in range(args.rep_num):
        if optimizer == 'smac':
            _optimizer = SMACOptimizer(evaluator, cs, evaluation_limit=runcount_limit, output_dir='logs', seed=i)
        elif optimizer == 'psmac':
            _optimizer = PSMACOptimizer(evaluator, cs, args.n, evaluation_limit=runcount_limit, output_dir='logs',
                                        trials_per_iter=args.trial, seed=i)
        print(perf)
        perf += _optimizer.run()
    print("time:" + str((time.time() - start_time) / args.rep_num))
    print(perf / args.rep_num)


if __name__ == "__main__":
    args = parser.parse_args()
    conduct_hpo(optimizer=args.optimizer, dataset=args.dataset, classifier_id=args.algo,
                runcount_limit=args.runcount_limit)

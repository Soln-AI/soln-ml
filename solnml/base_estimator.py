import os
from solnml.automl import AutoML
from solnml.components.metrics.metric import get_metric
from solnml.components.feature_engineering.transformation_graph import DataNode


class BaseEstimator(object):
    def __init__(
            self,
            time_limit=300,
            amount_of_resource=None,
            metric='acc',
            include_algorithms=None,
            ensemble_method='ensemble_selection',
            ensemble_size=50,
            per_run_time_limit=150,
            random_state=1,
            n_jobs=1,
            evaluation='holdout',
            output_dir="/tmp/"):
        self.metric = metric
        self.task_type = None
        self.time_limit = time_limit
        self.amount_of_resource = amount_of_resource
        self.include_algorithms = include_algorithms
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.per_run_time_limit = per_run_time_limit
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.evaluation = evaluation
        self.output_dir = output_dir
        self._ml_engine = None
        # Create output directory.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def build_engine(self):
        """Build AutoML controller"""
        engine = self.get_automl()(
            task_type=self.task_type,
            metric=self.metric,
            time_limit=self.time_limit,
            amount_of_resource=self.amount_of_resource,
            include_algorithms=self.include_algorithms,
            ensemble_method=self.ensemble_method,
            ensemble_size=self.ensemble_size,
            per_run_time_limit=self.per_run_time_limit,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            evaluation=self.evaluation,
            output_dir=self.output_dir
        )
        return engine

    def fit(self, data: DataNode):
        assert data is not None and isinstance(data, DataNode)
        self._ml_engine = self.build_engine()
        self._ml_engine.fit(data)
        return self

    def predict(self, X: DataNode, batch_size=None, n_jobs=1):
        return self._ml_engine.predict(X)

    def score(self, data: DataNode):
        return self._ml_engine.score(data)

    def refit(self):
        return self._ml_engine.refit()

    def predict_proba(self, X: DataNode, batch_size=None, n_jobs=1):
        return self._ml_engine.predict_proba(X)

    def get_automl(self):
        return AutoML

    def show_info(self):
        raise NotImplementedError()

    @property
    def best_config(self):
        return self._ml_engine.best_config

    @property
    def best_algo_id(self):
        return self._ml_engine.best_algo_id

    @property
    def best_perf(self):
        return self._ml_engine.best_perf

    def get_best_node_path(self):
        return self._ml_engine.fe_optimizer.get_pipeline(self._ml_engine.best_data_node)

    def get_ens_model_info(self):
        return self._ml_engine.get_ens_model_info()

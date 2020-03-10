from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter, UniformIntegerHyperparameter
from automlToolkit.components.feature_engineering.transformations.base_transformer import *
from automlToolkit.components.utils.configspace_utils import check_for_bool


# TODO: Select top-k features for transformer
# TODO: By what?
class PolynomialTransformationRegression(Transformer):
    def __init__(self, degree=2, interaction_only='True', include_bias='False', random_state=None):
        super().__init__("polynomial", 32)
        self.input_type = [DISCRETE, NUMERICAL]
        self.compound_mode = 'concatenate'

        self.output_type = NUMERICAL
        self.degree = degree
        if self.degree == 2:
            self.topn = 20
        elif self.degree == 3:
            self.topn = 10
        else:
            self.topn = 5
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.random_state = random_state
        self.select_model = None

    @ease_trans
    def operate(self, input_datanode, target_fields):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.feature_selection import SelectKBest, f_regression

        X, y = input_datanode.data
        X_new = X[:, target_fields]
        ori_length = X_new.shape[1]

        # # Skip high-dimensional features.
        # if X_new.shape[1] > 100:
        #     return X_new.copy()
        if not self.select_model:
            if ori_length < self.topn:
                k = 'all'
            else:
                k = self.topn
            self.select_model = SelectKBest(score_func=f_regression, k=k)
            self.select_model.fit(X_new, y)

        X_new = self.select_model.transform(X_new)
        selected_length = X_new.shape[1]

        if not self.model:
            self.degree = int(self.degree)
            self.interaction_only = check_for_bool(self.interaction_only)
            self.include_bias = check_for_bool(self.include_bias)

            self.model = PolynomialFeatures(
                degree=self.degree, interaction_only=self.interaction_only,
                include_bias=self.include_bias)
            self.model.fit(X_new)

        _X = self.model.transform(X_new)
        if selected_length == 1:
            _X = _X[:, 1:]
        else:
            _X = _X[:, selected_length + 1:]

        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        degree = UniformIntegerHyperparameter("degree", 2, 4, default_value=2)
        interaction_only = UnParametrizedHyperparameter("interaction_only", "True")
        include_bias = UnParametrizedHyperparameter("include_bias", "False")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([degree, interaction_only, include_bias])

        return cs

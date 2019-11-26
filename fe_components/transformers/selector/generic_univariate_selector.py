from fe_components.transformers.base_transformer import *


class GenericUnivariateSelector(Transformer):
    def __init__(self, param='chi2', feature_left=0.5):
        super().__init__("generic_univariate_selector", 6)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.params = param
        self.optional_params = ['chi2', 'f_classif']
        if param == 'chi2':
            from sklearn.feature_selection import chi2
            self.call_param = chi2
        elif param == 'f_classif':
            from sklearn.feature_selection import f_classif
            self.call_param = f_classif
        else:
            raise ValueError("Unknown score function %s!" % str(param))
        assert feature_left < 1.0
        self.feature_left = feature_left
        self.mode = 'fpr'

    def operate(self, input_datanode, target_fields=None):
        from sklearn.feature_selection import GenericUnivariateSelect

        feature_types = input_datanode.feature_types
        X, y = input_datanode.data
        if target_fields is None:
            target_fields = collect_fields(feature_types, self.input_type)
        X_new = X[:, target_fields]

        n_fields = len(feature_types)
        irrevalent_fields = list(range(n_fields))
        for field_id in target_fields:
            irrevalent_fields.remove(field_id)

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.params == 'chi2':
            X_new[X_new < 0] = 0.0

        if self.model is None:
            self.model = GenericUnivariateSelect(score_func=self.call_param, param=self.feature_left, mode=self.mode)
            self.model.fit(X_new, y)

        _X = self.model.transform(X_new)
        is_selected = self.model.get_support()

        irrevalent_types = [feature_types[idx] for idx in irrevalent_fields]
        selected_types = [feature_types[idx] for idx in target_fields if is_selected[idx]]
        selected_types.extend(irrevalent_types)

        new_X = np.hstack((_X, X[:, irrevalent_fields]))
        new_feature_types = selected_types
        output_datanode = DataNode((new_X, y), new_feature_types, input_datanode.task_type)

        return output_datanode
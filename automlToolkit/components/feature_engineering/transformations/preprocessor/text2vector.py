import numpy as np
from automlToolkit.components.feature_engineering.transformations.base_transformer import *
from automlToolkit.components.utils.text_util import *


class Text2VectorTransformation(Transformer):
    def __init__(self, param=None):
        super().__init__("text2vector", 0)
        self.params = param
        self.input_type = [TEXT]
        self.embedding_dict = build_embeddings_index()

    def operate(self, input_datanode, target_fields=None):
        pass

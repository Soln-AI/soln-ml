# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np

sys.path.append(os.getcwd())
from automlToolkit.components.feature_engineering.transformations.preprocessor.text2vector import *

print(load_text_embeddings(['This is a test, but fuck you!', 'I love you!'], method='weighted'))

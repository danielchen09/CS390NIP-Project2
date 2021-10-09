import random
import numpy as np
from Config import *
from typing import Any

class Model:
    def __init__(self, input_size: Any, output_size: int, config=None):
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        if config is None:
            self.config = DEFAULT_CONFIG

    def train(self, x_train, y_train):
        return self

    def predict(self, x):
        pass


class GuesserModel(Model):
    def predict(self, x):
        ans = []
        for entry in x:
            pred = [0] * self.output_size
            pred[random.randint(0, 9)] = 1
            ans.append(pred)
        return np.array(ans)

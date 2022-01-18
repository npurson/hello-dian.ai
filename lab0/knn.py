import numpy as np
from tqdm import tqdm


class Knn(object):

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):

        # TODO Predict the label of X by
        # the k nearest neighbors.

        ...

        # End of todo

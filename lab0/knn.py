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

        y_pred = []
        for x in tqdm(X):
            dist = [np.sum((xi - x) ** 2) for xi in self.X]
            topk = self.y[np.argsort(dist)[:self.k]]  # the top k nearest ys
            y_pred.append(np.argmax(np.bincount(topk)))  # the most frequent label in topk
        return y_pred

        # End of todo

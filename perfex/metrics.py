import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score


class Metric(ABC):
    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_min_samples_leaf(self):
        pass

    @abstractmethod
    def is_valid_dataset(self, X, y, y_pred, y_pred_proba):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class MetricAccuracy(Metric):
    def __init__(self, min_samples_leaf=100):
        self.min_samples_leaf = min_samples_leaf

    def get_name(self):
        return "accuracy"

    def get_min_samples_leaf(self):
        return self.min_samples_leaf

    def is_valid_dataset(self, X, y, y_pred, y_pred_proba):
        return len(X) > 0

    def evaluate(self, X, y, y_pred, y_pred_proba):
        if y is None:
            raise Exception("For the metric accuracy labels must be provided in y")
        return accuracy_score(y, y_pred)


class MetricPrecision(Metric):
    def __init__(
        self, selected_class, min_samples_leaf=100, min_samples_selected_class=50
    ):
        self.selected_class = selected_class
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_selected_class = min_samples_selected_class

    def get_name(self):
        return "precision for class {}".format(self.selected_class)

    def get_min_samples_leaf(self):
        return self.min_samples_leaf

    def is_valid_dataset(self, X, y, y_pred, y_pred_proba):
        return (
            len(X) > 0
            and np.sum(y_pred == self.selected_class) >= self.min_samples_selected_class
        )

    def evaluate(self, X, y, y_pred, y_pred_proba):
        if y is None:
            raise Exception("For the metric accuracy labels must be provided in y")
        return precision_score(y, y_pred, average=None, zero_division=0)[
            self.selected_class
        ]

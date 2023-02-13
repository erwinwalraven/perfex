import numpy as np
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier

from perfex import PERFEX
from perfex.metrics import MetricAccuracy

# create dataset and classifier
centers = np.array([[10, 10], [20, 12], [15, 15]])
X, y = make_blobs(
    n_samples=2000, centers=centers, n_features=2, cluster_std=3, random_state=1
)
feature_names = ["x", "y"]

# create model
dt = DecisionTreeClassifier(max_depth=4, random_state=0)
dt.fit(X, y)

metric = MetricAccuracy()
# metric = MetricPrecision(1)
perfex = PERFEX(
    dt,
    num_features=2,
    num_classes=3,
    feature_names=feature_names,
    class_labels=[0, 1, 2],
    max_depth=5,
    features_numerical=[0, 1],
    features_categorical=[],
    features_in_explanation=[],
    evaluate_stepwise=50,
)
perfex.fit(metric, X, y)
perfex.print_explanation()
print(perfex.predict_metric_value(X))
perfex.print_explanation_datapoint(np.array([12, 12]))

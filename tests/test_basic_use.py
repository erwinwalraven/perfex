import numpy as np
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from perfex import PERFEX
from perfex.metrics import MetricAccuracy


def test_decision_tree_accuracy():
    # create dataset and classifier
    centers = np.array([[10, 10], [20, 12], [15, 15]])
    X, y = make_blobs(
        n_samples=2000, centers=centers, n_features=2, cluster_std=3, random_state=1
    )
    feature_names = ["x", "y"]

    # create model
    dt = DecisionTreeClassifier(max_depth=4, random_state=0)
    dt.fit(X, y)

    # use perfex
    metric = MetricAccuracy()
    perfex = PERFEX(
        dt,
        num_features=2,
        num_classes=3,
        feature_names=feature_names,
        class_labels=[0, 1, 2],
        max_depth=5,
        features_numerical=[0, 1],
        evaluate_stepwise=50,
    )
    perfex.fit(metric, X, y)

    assert len(perfex.get_clusters()) == 8
    assert abs(perfex.predict_metric_value(np.array([[12, 12]]))[0] - 0.69293756) < 1e-5

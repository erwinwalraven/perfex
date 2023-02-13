import numpy as np
from sklearn.utils.validation import check_array
from perfex.metrics import Metric


class TreeNode:
    def __init__(
        self,
        X,
        y,
        y_pred,
        y_pred_proba,
        indicator_rows,
        metric,
        max_depth,
        map_feature_value_thresholds,
        features_categorical,
        features_in_explanation=[],
        parent=None,
        depth=0,
        decision_path=[],
        min_split_score_difference=0.05,
    ):
        assert isinstance(metric, Metric)
        assert max_depth > 0
        assert isinstance(map_feature_value_thresholds, dict)
        assert isinstance(features_categorical, list)
        assert isinstance(features_in_explanation, list)
        assert depth >= 0
        self.metric = metric
        self.max_depth = max_depth
        self.map_feature_value_thresholds = map_feature_value_thresholds
        self.features_categorical = features_categorical
        self.features_in_explanation = features_in_explanation
        self.min_split_score_difference = min_split_score_difference

        # original
        self.X = X
        self.y = y
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.indicator_rows = indicator_rows

        # evaluate value for this node
        self.value = self.metric.evaluate(
            self.X[indicator_rows],
            self.y[indicator_rows],
            self.y_pred[indicator_rows],
            self.y_pred_proba[indicator_rows],
        )

        # variables for creating tree structure
        self.parent = parent
        self.depth = depth
        self.feature_index = None
        self.split_threshold = None
        self.child_left = None
        self.child_right = None
        self.decision_path = decision_path

        # split current node if max depth has not been reached
        if self.depth < self.max_depth:
            self._split()

    def get_depth(self):
        if self.child_left is None:
            return 1
        else:
            return max(self.child_right.get_depth(), self.child_left.get_depth()) + 1

    def is_leaf(self):
        return self.child_left is None and self.child_right is None

    def _split(self):
        split_feature, split_threshold = self._determine_best_split()
        if split_feature != -1:
            self.feature_index = split_feature
            self.split_threshold = split_threshold
            indic_left = None
            indic_right = None
            if (
                len(self.features_categorical) > 0
                and split_feature in self.features_categorical
            ):
                indic_left = (
                    self.X[:, split_feature] == split_threshold
                ) * self.indicator_rows
                indic_right = np.logical_not(indic_left) * self.indicator_rows
                decision_path_left = self.decision_path.copy() + [
                    (split_feature, "==", split_threshold)
                ]
                decision_path_right = self.decision_path.copy() + [
                    (split_feature, "!=", split_threshold)
                ]
            else:
                indic_left = (
                    self.X[:, split_feature] <= split_threshold
                ) * self.indicator_rows
                indic_right = np.logical_not(indic_left) * self.indicator_rows
                decision_path_left = self.decision_path.copy() + [
                    (split_feature, "<=", split_threshold)
                ]
                decision_path_right = self.decision_path.copy() + [
                    (split_feature, ">", split_threshold)
                ]
            assert not indic_left is None and not indic_right is None
            self.child_left = TreeNode(
                self.X,
                self.y,
                self.y_pred,
                self.y_pred_proba,
                indic_left,
                self.metric,
                self.max_depth,
                self.map_feature_value_thresholds,
                self.features_categorical,
                self.features_in_explanation,
                parent=self,
                depth=self.depth + 1,
                decision_path=decision_path_left,
            )
            self.child_right = TreeNode(
                self.X,
                self.y,
                self.y_pred,
                self.y_pred_proba,
                indic_right,
                self.metric,
                self.max_depth,
                self.map_feature_value_thresholds,
                self.features_categorical,
                self.features_in_explanation,
                parent=self,
                depth=self.depth + 1,
                decision_path=decision_path_right,
            )

    def _determine_best_split(self):
        # this function enumerates all possible conditions to split the data in X, and returns best feature index and threshold
        best_split_score = 0
        best_split_feature = -1
        best_split_threshold = 0.0
        num_features = self.X.shape[1]
        for feature_index in range(num_features):
            if (
                len(self.features_in_explanation) > 0
                and feature_index not in self.features_in_explanation
            ):
                continue
            thresholds = self.map_feature_value_thresholds[feature_index]
            for j in range(len(thresholds)):
                # decide which datapoints belong to left and right
                candidate_threshold = thresholds[j]
                indic_left = None
                indic_right = None
                if (
                    len(self.features_categorical) > 0
                    and feature_index in self.features_categorical
                ):
                    indic_left = (
                        self.X[:, feature_index] == candidate_threshold
                    ) * self.indicator_rows
                    indic_right = np.logical_not(indic_left) * self.indicator_rows
                else:
                    indic_left = (
                        self.X[:, feature_index] <= candidate_threshold
                    ) * self.indicator_rows
                    indic_right = np.logical_not(indic_left) * self.indicator_rows
                assert not indic_left is None and not indic_right is None

                # make subsets
                X_left = self.X[indic_left]
                X_right = self.X[indic_right]
                y_left = self.y[indic_left]
                y_right = self.y[indic_right]
                y_pred_left = self.y_pred[indic_left]
                y_pred_right = self.y_pred[indic_right]
                y_pred_proba_left = self.y_pred_proba[indic_left]
                y_pred_proba_right = self.y_pred_proba[indic_right]

                # for this split the following conditions should be checked
                datapoints_both_left_right = np.any(indic_left) and np.any(indic_right)
                min_samples_leaf = self.metric.get_min_samples_leaf()
                min_samples_leaf_both_left_right = (
                    np.sum(indic_left) >= min_samples_leaf
                    and np.sum(indic_right) >= min_samples_leaf
                )
                dataset_left_valid = self.metric.is_valid_dataset(
                    X_left, y_left, y_pred_left, y_pred_proba_left
                )
                dataset_right_valid = self.metric.is_valid_dataset(
                    X_right, y_right, y_pred_right, y_pred_proba_right
                )

                # if all conditions hold, we check whether this split is better than the current best
                if (
                    datapoints_both_left_right
                    and min_samples_leaf_both_left_right
                    and dataset_left_valid
                    and dataset_right_valid
                ):
                    score_left = self.metric.evaluate(
                        X_left, y_left, y_pred_left, y_pred_proba_left
                    )
                    score_right = self.metric.evaluate(
                        X_right, y_right, y_pred_right, y_pred_proba_right
                    )
                    split_score = np.abs(score_left - score_right)
                    if (
                        split_score > best_split_score
                        and split_score >= self.min_split_score_difference
                    ):
                        best_split_score = split_score
                        best_split_feature = feature_index
                        best_split_threshold = candidate_threshold
        return best_split_feature, best_split_threshold

    def get_num_leafs(self):
        nodes = self.get_node_list()
        num_leafs = 0
        for node in nodes:
            if node.is_leaf():
                num_leafs += 1
        return num_leafs

    def get_node_list(self):
        return self._get_node_list([])

    def _get_node_list(self, nodes):
        # returns list containing all nodes in the tree for which this node is the root
        if self.is_leaf():
            nodes.append(self)
        else:
            self.child_left._get_node_list(nodes)
            self.child_right._get_node_list(nodes)
        return nodes

    def predict(self, X):
        # this function takes dataset as input and returns the leaf value for each datapoint
        # the interpretation of this value depends on the chosen metric

        # create array with True for each entry in X
        idx_indicator = np.ones(len(X)).astype(bool)

        # get a list which, for each leaf, defines which datapoints belong to that leaf, and the corresponding node value
        leaf_datapoints = self._predict(X.copy(), idx_indicator, [])

        # fill array with predictions
        predictions = np.zeros(len(X))
        datapoints_done = np.zeros(len(X)).astype(bool)

        for leaf in leaf_datapoints:
            idx_indicator = leaf[0]
            leaf_value = leaf[1]
            predictions[idx_indicator] = leaf_value
            datapoints_done[idx_indicator] = True

        # check whether values for all datapoints have been set, and return
        assert np.sum(datapoints_done) == len(X)
        return predictions

    def _predict(self, X, idx_indicator, leaf_datapoints):
        # idx_indicator[i] is true if X[i] still belongs to the current subtree
        assert len(X) == len(idx_indicator)
        if self.is_leaf():
            leaf_datapoints.append((idx_indicator, self.value))
        else:
            # use split condition to select datapoints belonging to both subtrees
            # perform logical and with idx_indicator to filter out datapoints that do not belong to current subtree
            idx_indicator_left = None
            idx_indicator_right = None
            if (
                len(self.features_categorical) > 0
                and self.feature_index in self.features_categorical
            ):
                idx_indicator_left = (
                    X[:, self.feature_index] == self.split_threshold
                ) * idx_indicator
                idx_indicator_right = np.logical_not(idx_indicator_left) * idx_indicator
            else:
                idx_indicator_left = (
                    X[:, self.feature_index] <= self.split_threshold
                ) * idx_indicator
                idx_indicator_right = np.logical_not(idx_indicator_left) * idx_indicator
            assert not idx_indicator_left is None and not idx_indicator_right is None
            self.child_left._predict(X, idx_indicator_left.copy(), leaf_datapoints)
            self.child_right._predict(X, idx_indicator_right.copy(), leaf_datapoints)
        return leaf_datapoints


class PERFEX:
    def __init__(
        self,
        model,
        num_features,
        num_classes,
        feature_names,
        class_labels,
        max_depth,
        features_numerical,
        features_categorical,
        features_in_explanation=[],
        evaluate_stepwise=0,
    ):
        assert num_features > 0 and num_classes > 0
        assert len(class_labels) == num_classes
        assert isinstance(class_labels, list)
        assert max_depth > 0
        assert len(feature_names) == num_features
        assert isinstance(features_numerical, list)
        assert isinstance(features_categorical, list)
        assert isinstance(features_in_explanation, list)
        self.model = model
        self.num_features = num_features
        self.num_classes = num_classes
        self.feature_names = feature_names
        self.class_labels = class_labels
        self.max_depth = max_depth
        self.features_numerical = features_numerical
        self.features_categorical = features_categorical

        self.features_in_explanation = features_in_explanation
        self.evaluate_stepwise = evaluate_stepwise
        self.tree = None
        self.metric = None
        self.clusters = None
        self._check_model()

    def _check_model(self):
        if not hasattr(self.model, "predict_proba"):
            raise Exception("Model does not provide a predict_proba function")
        if not hasattr(self.model, "predict"):
            raise Exception("Model does not provide a predict function")

    def _determine_feature_thresholds(self, X, feature_index):
        if self.evaluate_stepwise == 0 or (
            self.evaluate_stepwise > 0 and feature_index not in self.features_numerical
        ):
            # return all unique feature values
            unique_feature_values = np.unique(X[:, feature_index])
            if len(unique_feature_values) > 50:
                print(
                    "Feature {} has more than 50 unique values. Consider using the evaluate_stepwise parameter.".format(
                        feature_index
                    )
                )
            return unique_feature_values
        else:
            # return a limited number of feature values in ascending order
            vals = X[:, feature_index]
            if len(vals) < self.evaluate_stepwise:
                return np.unique(X[:, feature_index])
            else:
                idx_selected = list(
                    np.arange(
                        0, len(vals), len(vals) / (self.evaluate_stepwise - 1)
                    ).astype(int)
                ) + [len(vals) - 1]
                return np.sort(vals)[idx_selected]

    def fit(self, metric, X, y=None):
        assert isinstance(metric, Metric)
        X = check_array(X, dtype=np.float32)
        if not y is None:
            y = check_array(y, ensure_2d=False, dtype=None)
            if X.shape[0] != len(y):
                raise Exception(
                    "Number of samples in X does not match number of labels in y"
                )

        if X.shape[1] != self.num_features:
            raise Exception(
                "Number of features in X does not match number of features in PERFEX"
            )

        if not hasattr(metric, "evaluate"):
            raise Exception("Metric does not provide an evaluate function")

        # determine feature value thresholds used while generating the tree
        map_feature_value_thresholds = {}
        for j in range(X.shape[1]):
            map_feature_value_thresholds[j] = self._determine_feature_thresholds(X, j)

        y_pred = self.model.predict(X)  # predicted labels
        y_pred_proba = self.model.predict_proba(X)  # class probabilities
        indicator_all_rows = np.ones((X.shape[0],)).astype(bool)
        self.metric = metric
        self.tree = TreeNode(
            X,
            y,
            y_pred,
            y_pred_proba,
            indicator_all_rows,
            metric,
            self.max_depth,
            map_feature_value_thresholds,
            features_categorical=self.features_categorical,
            features_in_explanation=self.features_in_explanation,
            parent=None,
            depth=0,
            decision_path=[],
        )

        # extract clusters from tree
        self.clusters = []
        for tree_node in self.tree.get_node_list():
            if tree_node.is_leaf():
                self.clusters.append(
                    (tree_node.indicator_rows, tree_node.decision_path, tree_node.value)
                )

    def get_clusters(self):
        # returns a list containing a tuple for each cluster
        # a tuple contains:
        # - indicator defining the datapoints in X belonging to the cluster
        # - all conditions defining the cluster
        # - metric value computed based on the datapoints in the cluster
        if self.tree is None:
            raise Exception("fit has not been called yet")

        clusters_ret = []
        for cluster in self.clusters:
            # determine lower and upper bounds on feature values
            feature_has_lb = [False for i in range(self.num_features)]
            feature_has_ub = [False for i in range(self.num_features)]
            feature_lb = [-np.inf for i in range(self.num_features)]
            feature_ub = [np.inf for i in range(self.num_features)]
            for condition in cluster[1]:
                feature_index = condition[0]
                if (
                    len(self.features_categorical) > 0
                    and feature_index in self.features_categorical
                ):
                    continue
                rhs = condition[2]
                if condition[1] == ">" and rhs > feature_lb[feature_index]:
                    feature_has_lb[feature_index] = True
                    feature_lb[feature_index] = rhs
                elif condition[1] == "<=" and rhs < feature_ub[feature_index]:
                    feature_has_ub[feature_index] = True
                    feature_ub[feature_index] = rhs

            # add conditions
            conditions = []
            for j in range(self.num_features):
                if (
                    len(self.features_categorical) > 0
                    and j in self.features_categorical
                ):
                    # append all conditions for this feature because it is numerical
                    for condition in cluster[1]:
                        feature_index = condition[0]
                        if feature_index == j:
                            conditions.append((j, condition[1], condition[2]))
                else:
                    if feature_has_lb[j]:
                        conditions.append((j, ">", feature_lb[j]))
                    if feature_has_ub[j]:
                        conditions.append((j, "<=", feature_ub[j]))

            clusters_ret.append((cluster[0], conditions, cluster[2]))

        return clusters_ret

    def predict_metric_value(self, X):
        # predicts the metric value for each datapoint in X
        if self.tree is None:
            raise Exception("fit has not been called yet")
        X = check_array(X, dtype=np.float32)
        if X.shape[1] != self.num_features:
            raise Exception(
                "Number of features in X does not match number of features in PERFEX"
            )
        return self.tree.predict(X)

    def print_explanation_datapoint(self, x):
        assert isinstance(x, np.ndarray)
        assert len(x) == self.num_features
        X = np.array([x])
        metric_value = self.tree.predict(X)[0]
        print(
            "The datapoint belongs to a cluster for which the metric '{}' evaluates to {}".format(
                self.metric.get_name(), metric_value
            )
        )

    def print_explanation(self):
        if self.tree is None:
            raise Exception("fit has not been called yet")

        # create a list with clusters sorted by their metric value
        clusters_sorted = [(cluster, cluster[2]) for cluster in self.get_clusters()]
        clusters_sorted.sort(key=lambda x: x[1])
        clusters_sorted = [cluster[0] for cluster in clusters_sorted]

        # print explanation
        for i in range(len(clusters_sorted)):
            cluster = clusters_sorted[i]
            num_datapoints = np.sum(cluster[0])

            print(
                "There are {} datapoints for which the following conditions hold:".format(
                    num_datapoints
                )
            )

            for condition in cluster[1]:
                print("", self.feature_names[condition[0]], condition[1], condition[2])

            print(
                "and for these datapoints the metric '{}' evaluates to {}".format(
                    self.metric.get_name(), cluster[2]
                )
            )
            if i < len(clusters_sorted) - 1:
                print()

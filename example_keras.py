import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_blobs
from perfex.model_wrappers import ModelWrapperKeras
from perfex import PERFEX
from perfex.metrics import MetricAccuracy

# create dataset and classifier
centers = np.array([[10, 10], [20, 12], [15, 15]])
X, y = make_blobs(
    n_samples=2000, centers=centers, n_features=2, cluster_std=3, random_state=1
)
feature_names = ["x", "y"]


encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


model = Sequential()
model.add(Dense(4, input_dim=2, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, dummy_y, epochs=30, batch_size=10, verbose=0)

class_labels = [0, 1, 2]
model_wrapper = ModelWrapperKeras(model, class_labels)


# use perfex
metric = MetricAccuracy()
# metric = MetricPrecision(1)
perfex = PERFEX(
    model_wrapper,
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

import torch
from torch.autograd import Variable
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_blobs
from perfex import PERFEX
from perfex.metrics import MetricAccuracy
from perfex.model_wrappers import ModelWrapperPyTorch

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


x_train = torch.from_numpy(X.astype(np.float32))
y_train = torch.from_numpy(dummy_y.astype(np.float32))

model = torch.nn.Sequential(
    torch.nn.Linear(2, 3, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(3, 3, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(3, 3, bias=True),
    torch.nn.ReLU(),
    torch.nn.Softmax(dim=1),
)

num_epoch = 1000
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(num_epoch):
    input = Variable(x_train)
    target = Variable(y_train)
    out = model(input)
    loss = loss_function(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# predicting
predict_proba_vals = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
predict_vals = np.argmax(predict_proba_vals, axis=1)


class_labels = [0, 1, 2]
model_wrapper = ModelWrapperPyTorch(model, class_labels)


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

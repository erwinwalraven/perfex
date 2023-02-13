import torch
import numpy as np


class ModelWrapperPyTorch:
    def __init__(self, model, class_labels):
        self.model = model
        self.class_labels_np = np.array(class_labels)

    def predict_proba(self, X):
        return self.model(torch.tensor(X, dtype=torch.float32)).detach().numpy()

    def predict(self, X):
        y_pred_proba = self.model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
        predicted_classes_idx = np.argmax(y_pred_proba, axis=1)
        return self.class_labels_np[predicted_classes_idx]

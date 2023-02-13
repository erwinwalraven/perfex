import numpy as np


class ModelWrapperKeras:
    def __init__(self, model, class_labels):
        self.model = model
        self.class_labels_np = np.array(class_labels)

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict(self, X):
        y_pred_proba = self.model.predict(X)
        predicted_classes_idx = np.argmax(y_pred_proba, axis=1)
        return self.class_labels_np[predicted_classes_idx]

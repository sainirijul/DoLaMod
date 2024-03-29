import os
import pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin


class GloveVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self._nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.concatenate([self._nlp(doc).vector.reshape(1, -1) for doc in X])


class ConceptsPrediction:
    def __init__(self):
        pass

    def predict_category(self, test_data_tokens, g):
        test_data = []
        for t in test_data_tokens:
            if type(t) != str:
                test_data.append(t.text)
            else:
                test_data.append(t.lower())

        dirname = os.path.dirname(__file__)

        filepath_ml_category = os.path.join(
            dirname, "../", "data", "trained_models", "mod_org_cat_9"
        )
        filepath_ml_type = os.path.join(
            dirname, "../", "data", "trained_models", "mod_org_type_9"
        )

        filepath_encoder_category = os.path.join(
            dirname, "../", "data", "trained_models", "encoder_category.pickle"
        )
        filepath_encoder_type = os.path.join(
            dirname, "../", "data", "trained_models", "encoder_type.pickle"
        )

        loaded_model_type = pickle.load(open(filepath_ml_type, "rb"))
        loaded_model_cat = pickle.load(open(filepath_ml_category, "rb"))

        y_test_pred_type = loaded_model_type.predict(g.transform(test_data))
        class_prob_type = loaded_model_type.predict_proba(g.transform(test_data))

        y_test_pred_cat = loaded_model_cat.predict(g.transform(test_data))
        class_prob_cat = loaded_model_cat.predict_proba(g.transform(test_data))

        with open(filepath_encoder_type, "rb") as file:
            encoder_type = pickle.load(file)

        with open(filepath_encoder_category, "rb") as file:
            encoder_category = pickle.load(file)

        Y_pred_label_type = list(encoder_type.inverse_transform(y_test_pred_type))
        Y_pred_label_cat = list(encoder_category.inverse_transform(y_test_pred_cat))

        return Y_pred_label_cat, Y_pred_label_type, class_prob_cat, class_prob_type

import os
import pickle

import numpy as np
import pandas as pd
import spacy
from sklearn import preprocessing, svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class GloveVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp, doc):
        self._nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.concatenate([self._nlp(doc).vector.reshape(1, -1) for doc in X])


class ConceptsPrediction:
    def __init__(self):
        pass

    def train_models(self, g):
        encoder_type = preprocessing.LabelEncoder()
        encoder_category = preprocessing.LabelEncoder()

        dirname = os.path.dirname(__file__)
        pathname_train_data = os.path.abspath(
            os.path.join(dirname, "../", "data", "training", "ml_data.csv")
        )
        train_data = pd.read_csv(pathname_train_data)
        train_data = train_data.sample(frac=1)
        train_data["category"] = np.where(train_data["category"] == "attribute", 1, 0)

        encoder_category.fit(train_data["category"])
        pickle.dump(
            encoder_category,
            open(
                os.path.abspath(
                    os.path.join(
                        dirname, "../", "data", "trained_models", "encoder_category"
                    )
                ),
                "wb",
            ),
        )

        # parameter_candidates = [
        #     {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        #     {'C': [1, 10, 100, 1000], 'gamma': [0.01,0.001,0.1], 'kernel': ['rbf']},
        #     {'C': [10], 'gamma': [0.001], 'kernel': ['rbf'], 'decision_function_shape': ['ovr'], 'break_ties' : [False]}
        #     ]

        parameter_candidates = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30],
            "min_samples_split": [2, 5, 10],
        }
        classifier = RandomForestClassifier()

        # Transform the series of text into vectors
        train_data["text_processed"] = train_data["text"].apply(lambda x: g(x))
        train_vectors = np.array(
            train_data["text_processed"].apply(lambda doc: doc.vector).tolist()
        )

        # Define your labels
        train_cat_labels = train_data["category"].values
        train_type_labels = train_data["type"].values

        # Define SVM and GridSearchCV
        # parameter_candidates = [{'C': [10], 'kernel': ['linear']}]
        # classifier = svm.SVC()
        svm_model_cat = GridSearchCV(classifier, parameter_candidates, cv=3)

        # Fit the model
        svm_model_cat.fit(train_vectors, train_cat_labels)

        final_model_cat = svm_model_cat.best_estimator_
        print(
            f"Best Training Performance (Cross-Validation Accuracy): {svm_model_cat.best_score_:.4f}"
        )

        filename_cat = os.path.abspath(
            os.path.join(dirname, "../", "data", "trained_models", "mod_org_cat_9")
        )
        pickle.dump(final_model_cat, open(filename_cat, "wb"))

        encoder_type.fit(train_data["type"])

        pickle.dump(
            encoder_type,
            open(
                os.path.abspath(
                    os.path.join(
                        dirname, "../", "data", "trained_models", "encoder_type"
                    )
                ),
                "wb",
            ),
        )

        Y_train = encoder_type.transform(train_type_labels)

        svm_model = GridSearchCV(classifier, parameter_candidates, cv=3)

        svm_model.fit(train_vectors, Y_train)

        final_model = svm_model.best_estimator_
        print(
            f"Best Training Performance (Cross-Validation Accuracy): {svm_model.best_score_:.4f}"
        )

        filename1 = os.path.abspath(
            os.path.join(dirname, "../", "data", "trained_models", "mod_org_type_9")
        )

        pickle.dump(final_model, open(filename1, "wb"))
        print("Training completed! Models have been saved.")

    def predict_category(self, test_data_tokens, g):
        test_data = []
        for t in test_data_tokens:
            if type(t) != str:
                test_data.append(t.text)
            else:
                test_data.append(t.lower())

        dirname = os.path.dirname(__file__)

        test_vectors = np.array([g(" ".join(tokens)).vector for tokens in test_data])

        filepath_ml_category = os.path.abspath(
            os.path.join(dirname, "../", "data", "trained_models", "mod_org_cat_9")
        )
        filepath_ml_type = os.path.abspath(
            os.path.join(dirname, "../", "data", "trained_models", "mod_org_type_9")
        )

        filepath_encoder_category = os.path.abspath(
            os.path.join(dirname, "../", "data", "trained_models", "encoder_category")
        )
        filepath_encoder_type = os.path.abspath(
            os.path.join(dirname, "../", "data", "trained_models", "encoder_type")
        )

        loaded_model_type = pickle.load(open(filepath_ml_type, "rb"))
        loaded_model_cat = pickle.load(open(filepath_ml_category, "rb"))

        y_test_pred_type = loaded_model_type.predict(test_vectors)

        y_test_pred_cat = loaded_model_cat.predict(test_vectors)

        with open(filepath_encoder_type, "rb") as file:
            encoder_type = pickle.load(file)

        with open(filepath_encoder_category, "rb") as file:
            encoder_category = pickle.load(file)

        Y_pred_label_type = list(encoder_type.inverse_transform(y_test_pred_type))
        Y_pred_label_cat = list(encoder_category.inverse_transform(y_test_pred_cat))

        return Y_pred_label_cat, Y_pred_label_type


if __name__ == "__main__":
    predictor = ConceptsPrediction()
    lm = spacy.load("en_core_web_lg")
    # predictor.train_models(lm)
    print(predictor.predict_category(["name"], lm))

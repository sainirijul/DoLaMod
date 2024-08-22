import os
import pickle

import numpy as np
import pandas as pd
import spacy
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


class ConceptsPrediction:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        pass

    def preprocess_text(self, text):
        doc = self.nlp(text.lower())
        return doc.vector

    def train_models(self):
        encoder_type = preprocessing.LabelEncoder()
        encoder_category = preprocessing.LabelEncoder()

        dirname = os.path.dirname(__file__)
        pathname_train_data = os.path.abspath(
            os.path.join(dirname, "../", "data", "training", "ml_data.csv")
        )
        train_data = pd.read_csv(pathname_train_data)
        train_data = train_data.sample(frac=1)

        # Encode category
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

        # Text preprocessing
        train_data["text_processed"] = train_data["text"].apply(self.preprocess_text)
        train_vectors = np.array(train_data["text_processed"].tolist())

        # Split data for category and type prediction
        train_vectors_cat = train_vectors
        train_cat_labels = train_data["category"].values

        train_vectors_type = train_vectors
        train_type_labels = train_data["type"].values

        # SMOTE for class imbalance in category prediction
        smote = SMOTE()
        train_vectors_cat, train_cat_labels = smote.fit_resample(
            train_vectors_cat, train_cat_labels
        )

        # RandomForest for Category Prediction
        param_grid_rf = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt"],
        }

        rf_classifier_cat = RandomForestClassifier()
        grid_rf_cat = GridSearchCV(
            rf_classifier_cat,
            param_grid_rf,
            cv=StratifiedKFold(n_splits=5),
            scoring="f1",
            verbose=1,
            n_jobs=-1,
        )
        grid_rf_cat.fit(train_vectors_cat, train_cat_labels)

        final_model_cat = grid_rf_cat.best_estimator_
        print(
            f"Best Training Performance (Cross-Validation Accuracy): {grid_rf_cat.best_score_:.4f}"
        )
        filename_cat = os.path.abspath(
            os.path.join(dirname, "../", "data", "trained_models", "mod_org_cat_9")
        )
        pickle.dump(final_model_cat, open(filename_cat, "wb"))

        # Encode types
        encoder_type.fit(train_type_labels)
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

        # GradientBoosting for Type Prediction
        param_grid_gb = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.1, 0.01, 0.001],
            "max_depth": [3, 5, 7],
        }

        gb_classifier_type = GradientBoostingClassifier()
        grid_gb_type = GridSearchCV(
            gb_classifier_type,
            param_grid_gb,
            cv=StratifiedKFold(n_splits=5),
            scoring="accuracy",
            verbose=1,
            n_jobs=-1,
        )
        grid_gb_type.fit(train_vectors_type, Y_train)

        final_model_type = grid_gb_type.best_estimator_
        print(
            f"Best Training Performance (Cross-Validation Accuracy): {grid_gb_type.best_score_:.4f}"
        )
        filename_type = os.path.abspath(
            os.path.join(dirname, "../", "data", "trained_models", "mod_org_type_9")
        )
        pickle.dump(final_model_type, open(filename_type, "wb"))

        print("Training completed! Models have been saved.")

    def predict_category(self, test_data_tokens, nlp):
        test_data = []
        for t in test_data_tokens:
            if type(t) != str:
                test_data.append(t.text)
            else:
                test_data.append(t.lower())

        dirname = os.path.dirname(__file__)

        # Convert test data to vectors
        test_vectors = np.array([nlp(" ".join(test_data)).vector])

        # Load the models and encoders
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

        with open(filepath_ml_category, "rb") as file:
            loaded_model_cat = pickle.load(file)

        with open(filepath_ml_type, "rb") as file:
            loaded_model_type = pickle.load(file)

        with open(filepath_encoder_category, "rb") as file:
            encoder_category = pickle.load(file)

        with open(filepath_encoder_type, "rb") as file:
            encoder_type = pickle.load(file)

        # Predict category and type
        y_test_pred_cat = loaded_model_cat.predict(test_vectors)
        y_test_pred_type = loaded_model_type.predict(test_vectors)

        # Decode predictions
        Y_pred_label_cat = encoder_category.inverse_transform(y_test_pred_cat)
        Y_pred_label_type = encoder_type.inverse_transform(y_test_pred_type)

        return Y_pred_label_cat[0], Y_pred_label_type[0]


if __name__ == "__main__":
    predictor = ConceptsPrediction()
    nlp = spacy.load("en_core_web_lg")
    # predictor.train_models()
    print("For concept User ", predictor.predict_category(["user"], nlp))
    print("\nFor concept name ", predictor.predict_category(["name"], nlp))

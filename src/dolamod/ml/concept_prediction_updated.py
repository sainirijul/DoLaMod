import os
import pickle

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ConceptsPrediction:
    def __init__(self):
        # Loading GloVe model (300-dimension vectors)
        glove_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "glove.6B.300d.txt")
        )
        self.glove_model = KeyedVectors.load_word2vec_format(
            glove_path, binary=False, no_header=True
        )

    def preprocess_text(self, text):
        # Tokenize text and get GloVe embeddings
        tokens = text.lower().split()
        embeddings = [
            self.glove_model[word] for word in tokens if word in self.glove_model
        ]

        if len(embeddings) > 0:
            return np.mean(embeddings, axis=0)  # Take the mean of all token embeddings
        else:
            return np.zeros(300)  # Return zero vector if no embeddings found

    def train_models(self):
        encoder_type = preprocessing.LabelEncoder()
        encoder_category = preprocessing.LabelEncoder()

        # Load training data
        dirname = os.path.dirname(__file__)
        pathname_train_data = os.path.abspath(
            os.path.join(dirname, "../", "data", "training", "ml_data.csv")
        )
        train_data = pd.read_csv(pathname_train_data)
        train_data = train_data.sample(frac=1)

        # Encode category and type labels
        encoder_category.fit(train_data["category"])
        encoder_type.fit(train_data["type"])

        # Save encoders
        with open(
            os.path.join(dirname, "../", "data", "trained_models", "encoder_category"),
            "wb",
        ) as f:
            pickle.dump(encoder_category, f)
        with open(
            os.path.join(dirname, "../", "data", "trained_models", "encoder_type"), "wb"
        ) as f:
            pickle.dump(encoder_type, f)

        # Preprocess text and convert to vectors
        train_data["text_processed"] = train_data["text"].apply(self.preprocess_text)
        train_vectors = np.array(train_data["text_processed"].tolist())

        # Separate features for category and type prediction
        train_vectors_cat = train_vectors
        train_cat_labels = encoder_category.transform(train_data["category"].values)

        train_vectors_type = train_vectors
        train_type_labels = encoder_type.transform(train_data["type"].values)

        # Use SMOTE to handle class imbalance
        smote = SMOTE()
        train_vectors_cat, train_cat_labels = smote.fit_resample(
            train_vectors_cat, train_cat_labels
        )

        # Define RandomForest for category prediction
        rf_classifier_cat = RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42
        )
        rf_classifier_cat.fit(train_vectors_cat, train_cat_labels)

        # Save category model
        with open(
            os.path.join(dirname, "../", "data", "trained_models", "model_category"),
            "wb",
        ) as f:
            pickle.dump(rf_classifier_cat, f)

        # Define GradientBoosting for type prediction
        gb_classifier_type = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.01, max_depth=5, random_state=42
        )
        gb_classifier_type.fit(train_vectors_type, train_type_labels)

        # Save type model
        with open(
            os.path.join(dirname, "../", "data", "trained_models", "model_type"), "wb"
        ) as f:
            pickle.dump(gb_classifier_type, f)

        print("Training completed and models saved!")

    def predict_category(self, text):
        dirname = os.path.dirname(__file__)

        # Preprocess text
        text_vector = np.array([self.preprocess_text(text)])

        # Load models and encoders
        with open(
            os.path.join(dirname, "../", "data", "trained_models", "model_category"),
            "rb",
        ) as f:
            model_cat = pickle.load(f)
        with open(
            os.path.join(dirname, "../", "data", "trained_models", "model_type"), "rb"
        ) as f:
            model_type = pickle.load(f)
        with open(
            os.path.join(dirname, "../", "data", "trained_models", "encoder_category"),
            "rb",
        ) as f:
            encoder_category = pickle.load(f)
        with open(
            os.path.join(dirname, "../", "data", "trained_models", "encoder_type"), "rb"
        ) as f:
            encoder_type = pickle.load(f)

        # Predict category and type
        pred_cat = model_cat.predict(text_vector)
        pred_type = model_type.predict(text_vector)

        return (
            encoder_category.inverse_transform(pred_cat)[0],
            encoder_type.inverse_transform(pred_type)[0],
        )


if __name__ == "__main__":
    predictor = ConceptsPrediction()
    # predictor.train_models()
    print(
        "Prediction for 'presentation time':",
        predictor.predict_category("presentation time"),
    )

    print("For concept User ", predictor.predict_category("user"))
    print("\nFor concept name ", predictor.predict_category("name"))
    print("\nFor concept University ", predictor.predict_category("University"))
    print("\nFor concept Department ", predictor.predict_category("Department"))
    print("\nFor concept email ", predictor.predict_category("Email"))
    print("\nFor concept number ", predictor.predict_category("number"))
    print("\nFor concept address ", predictor.predict_category("address"))
    print("\nFor concept ID ", predictor.predict_category("ID"))

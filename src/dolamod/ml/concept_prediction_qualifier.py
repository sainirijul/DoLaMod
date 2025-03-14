import os
import pickle

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


class ConceptsPrediction:
    def __init__(self):
        # Load GloVe model (300-dimension vectors)
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
        return np.mean(embeddings, axis=0) if embeddings else np.zeros(300)

    def preprocess_data(self, data):
        # Preprocess concept, qualifier, and context into embeddings
        data["concept_embedding"] = data["concept"].apply(self.preprocess_text)
        data["qualifier_embedding"] = data["qualifier"].apply(self.preprocess_text)
        data["context_embedding"] = data["context"].apply(self.preprocess_text)

        # Concatenate concept, qualifier, and context embeddings
        features = np.hstack(
            [
                data["concept_embedding"].tolist(),
                data["qualifier_embedding"].tolist(),
                data["context_embedding"].tolist(),
            ]
        )
        return features

    def train_models(self):
        # Load training data
        dirname = os.path.dirname(__file__)
        pathname_train_data = os.path.abspath(
            os.path.join(
                dirname, "../", "data", "training", "expanded_corrected_dataset.csv"
            )
        )
        train_data = pd.read_csv(pathname_train_data)
        train_data_filtered = train_data[
            (train_data["type"] != "class") | (train_data["type"] != "none")
        ].reset_index(drop=True)
        train_data_filtered = train_data_filtered.sample(frac=1)

        # Encode type labels
        encoder_type = preprocessing.LabelEncoder()
        train_data_filtered["type_encoded"] = encoder_type.fit_transform(
            train_data_filtered["type"]
        )

        # Preprocess data into feature vectors
        train_vectors = self.preprocess_data(train_data_filtered)
        train_labels = train_data_filtered["type_encoded"]

        # Handle class imbalance using SMOTE
        smote = SMOTE()
        train_vectors, train_labels = smote.fit_resample(train_vectors, train_labels)

        # Train RandomForest for type prediction
        rf_classifier = RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42
        )
        rf_classifier.fit(train_vectors, train_labels)

        # Save type model
        with open(
            os.path.join(
                dirname, "../", "data", "trained_models", "model_type_qualifier"
            ),
            "wb",
        ) as f:
            pickle.dump(rf_classifier, f)

        with open(
            os.path.join(
                dirname, "../", "data", "trained_models", "encoder_type_qualifier"
            ),
            "wb",
        ) as f:
            pickle.dump(encoder_type, f)

        print("Training completed and model saved!")

    def predict_type(self, concept, qualifier, context):
        # Preprocess input into embeddings
        concept_embedding = self.preprocess_text(concept)
        qualifier_embedding = self.preprocess_text(qualifier)
        context_embedding = self.preprocess_text(context)

        # Concatenate all features
        input_vector = np.hstack(
            [concept_embedding, qualifier_embedding, context_embedding]
        )
        dirname = os.path.dirname(__file__)
        with open(
            os.path.join(
                dirname, "../", "data", "trained_models", "model_type_qualifier"
            ),
            "rb",
        ) as f:
            model_type = pickle.load(f)
        with open(
            os.path.join(
                dirname, "../", "data", "trained_models", "encoder_type_qualifier"
            ),
            "rb",
        ) as f:
            encoder_type = pickle.load(f)
        data_reshaped = np.array(input_vector).reshape(1, -1)
        pred_type = model_type.predict(data_reshaped)

        return encoder_type.inverse_transform(pred_type)[0]

    def predict_category_with_probability(self, concept, context="", qualifier=""):
        dirname = os.path.dirname(__file__)

        # Preprocess input into embeddings
        concept_embedding = self.preprocess_text(concept)
        qualifier_embedding = self.preprocess_text(qualifier)
        context_embedding = self.preprocess_text(context)

        # Concatenate all features
        input_vector = np.hstack(
            [concept_embedding, qualifier_embedding, context_embedding]
        )
        with open(
            os.path.join(
                dirname, "../", "data", "trained_models", "model_type_qualifier"
            ),
            "rb",
        ) as f:
            model_type = pickle.load(f)
        with open(
            os.path.join(
                dirname, "../", "data", "trained_models", "encoder_type_qualifier"
            ),
            "rb",
        ) as f:
            encoder_type = pickle.load(f)

        data_reshaped = np.array(input_vector).reshape(1, -1)
        pred_type = model_type.predict(data_reshaped)

        return encoder_type.inverse_transform(pred_type)[0]


if __name__ == "__main__":
    predictor = ConceptsPrediction()
    predictor.train_models()
    print(predictor.predict_type("number", "passport", "user"))

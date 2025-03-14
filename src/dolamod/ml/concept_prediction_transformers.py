import os
import pickle

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from transformers import DistilBertModel, DistilBertTokenizer


class ConceptPrediction:
    def __init__(self):
        # Load DistilBERT tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def preprocess_text(self, text):
        # Tokenize and get embeddings from DistilBERT
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def preprocess_data(self, data):
        # Preprocess concept, qualifier, and context into embeddings
        data["concept_embedding"] = data["concept"].apply(self.preprocess_text)
        data["qualifier_embedding"] = data["qualifier"].apply(
            lambda x: self.preprocess_text(x) if pd.notnull(x) else np.zeros(300)
        )
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

    def train_model(self, train_data_path, model_save_path):
        # Load training data
        train_data = pd.read_csv(train_data_path)

        # Encode type labels
        encoder_type = preprocessing.LabelEncoder()
        train_data["type_encoded"] = encoder_type.fit_transform(train_data["type"])

        # Preprocess data into feature vectors
        train_vectors = self.preprocess_data(train_data)
        train_labels = train_data["type_encoded"]

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        train_vectors, train_labels = smote.fit_resample(train_vectors, train_labels)

        # Train MLP Classifier
        mlp_classifier = MLPClassifier(
            hidden_layer_sizes=(512, 256), max_iter=300, random_state=42
        )
        mlp_classifier.fit(train_vectors, train_labels)

        # Save trained model and label encoder
        with open(os.path.join(model_save_path, "mlp_model.pkl"), "wb") as f:
            pickle.dump(mlp_classifier, f)
        with open(os.path.join(model_save_path, "mlp_encoder.pkl"), "wb") as f:
            pickle.dump(encoder_type, f)

        print("Training completed and model saved!")

    def predict_category_with_probability(
        self,
        concept,
        qualifier,
        context,
        model_save_path="/home/rijul/Desktop/ShradhaSaburi/Learning/2024/Engineering/DoLaMod/src/dolamod/data/trained_models/",
    ):
        # Preprocess input into embeddings
        concept_embedding = self.preprocess_text(concept)
        qualifier_embedding = (
            self.preprocess_text(qualifier) if qualifier else np.zeros(768)
        )
        context_embedding = self.preprocess_text(context)

        # Concatenate embeddings
        input_vector = np.hstack(
            [concept_embedding, qualifier_embedding, context_embedding]
        )

        # Load trained model and encoder
        with open(os.path.join(model_save_path, "mlp_model.pkl"), "rb") as f:
            mlp_model = pickle.load(f)
        with open(os.path.join(model_save_path, "mlp_encoder.pkl"), "rb") as f:
            encoder = pickle.load(f)

        # Predict type
        pred_type = mlp_model.predict([input_vector])
        return encoder.inverse_transform(pred_type)[0]


if __name__ == "__main__":
    predictor = ConceptPrediction()
    train_data_path = "../data/training/meaningful_training_data.csv"
    model_save_path = "../data/trained_models/"

    predictor.train_model(train_data_path, model_save_path)
    print(
        predictor.predict_category_with_probability(
            "area", "urban", "geography", model_save_path
        )
    )

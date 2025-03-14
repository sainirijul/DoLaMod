import os
import pickle

import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from transformers import BertModel, BertTokenizer

# Load pre-trained BERT model and tokenizer
# TODO uncomment for using bert model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


class ConceptsPrediction:
    def __init__(self):
        # Loading GloVe model (300-dimension vectors)
        glove_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "glove.6B.300d.txt")
        )
        self.glove_model = KeyedVectors.load_word2vec_format(
            glove_path, binary=False, no_header=True
        )

    def preprocess_text_bert(self, text):
        # Tokenize the input text and convert tokens to input IDs
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Get the hidden states from BERT (output includes embeddings for all tokens)
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the embeddings (use the hidden states from the last layer)
        embeddings = (
            outputs.last_hidden_state
        )  # Shape: (batch_size, sequence_length, hidden_size)

        # You can average or pool the token embeddings for the sentence representation if needed
        sentence_embedding = embeddings.mean(dim=1)  # Averaging over token embeddings
        # print(sentence_embedding)
        return sentence_embedding[0].numpy()

    def preprocess_text(self, text):
        # To handle empty context
        if type(text) == float:
            return np.zeros(300)

        # Tokenize text and get GloVe embeddings
        tokens = text.lower().split()
        embeddings = [
            self.glove_model[word] for word in tokens if word in self.glove_model
        ]

        # if len(embeddings) > 0:
        #     concatenated_embedding = np.concatenate(embeddings)
        # else:
        #     # If the list is empty, create an empty array
        #     concatenated_embedding = np.array([])
        # # If the concatenated embedding length is less than 1200, append zeros
        # if len(concatenated_embedding) < 1200:
        #     concatenated_embedding = np.pad(concatenated_embedding, (0, 1200 - len(concatenated_embedding)), 'constant')
        #
        # return concatenated_embedding
        if len(embeddings) > 0:
            return np.mean(embeddings, axis=0)  # Take the mean of all token embeddings
        else:
            return np.zeros(300)  # Return zero vector if no embeddings found

    def train_models(self):
        encoder_type = preprocessing.LabelEncoder()

        # TODO : Change these directories for your system
        data_directory = "data/training/expended_corrected_dataset.csv"
        model_directory = "data/trained_models/gpt_data_bert"

        # Load training data
        dirname = os.path.dirname(__file__)
        pathname_train_data = os.path.abspath(os.path.join(dirname, data_directory))
        train_data = pd.read_csv(pathname_train_data)
        train_data = train_data.sample(frac=1)

        encoder_type.fit(train_data["type"])

        with open(os.path.join(dirname, model_directory, "encoder_type"), "wb") as f:
            pickle.dump(encoder_type, f)

        # Preprocess text and convert to vectors
        train_data_embeddings = pd.DataFrame(
            columns=["qualifier", "concept", "context"]
        )

        # TODO Use preprocess_text_bert() if you want to use bert embeddings
        train_data_embeddings["qualifier"] = train_data["qualifier"].apply(
            self.preprocess_text_bert
        )
        train_data_embeddings["concept"] = train_data["concept"].apply(
            self.preprocess_text_bert
        )
        train_data_embeddings["context"] = train_data["context"].apply(
            self.preprocess_text_bert
        )

        train_data_embeddings["combined"] = train_data_embeddings.apply(
            lambda row: np.concatenate(
                [row["qualifier"], row["concept"], row["context"]]
            ),
            axis=1,
        )

        train_vectors = np.array(train_data_embeddings["combined"].tolist())

        train_vectors_type = train_vectors
        train_type_labels = encoder_type.transform(train_data["type"].values)

        # Define GradientBoosting for type prediction
        gb_classifier_type = GradientBoostingClassifier(
            n_estimators=30, learning_rate=0.01, max_depth=5, random_state=42
        )
        # gb_classifier_type = RandomForestClassifier(n_estimators=100, criterion='log_loss', max_depth=5,
        # warm_start=True)
        gb_classifier_type.fit(train_vectors_type, train_type_labels)

        # Save type model
        with open(os.path.join(dirname, model_directory, "model_type"), "wb") as f:
            pickle.dump(gb_classifier_type, f)

        print("Training completed and models saved!")

    def predict_category(self, text):
        dirname = os.path.dirname(__file__)

        model_directory = "data/training/synthetic_data_bert"

        # Preprocess text
        text_vector = np.array([self.preprocess_text(text)])

        with open(os.path.join(dirname, model_directory, "model_type"), "rb") as f:
            model_type = pickle.load(f)
        with open(os.path.join(dirname, model_directory, "encoder_type"), "rb") as f:
            encoder_type = pickle.load(f)

        pred_type = model_type.predict(text_vector)

        return (
            "NA",
            encoder_type.inverse_transform(pred_type)[0],
        )

    def predict_category_with_probability(self, concept, context="", qualifier=""):
        dirname = os.path.dirname(__file__)

        model_directory = "data/trained_models/gpt_data_bert"

        # Preprocess text
        # TODO Use preprocess_text_bert() if you want to use bert embeddings
        concept_vector = np.array([self.preprocess_text_bert(concept)])
        context_vector = np.array([self.preprocess_text_bert(context)])
        qualifier_vector = np.array([self.preprocess_text_bert(qualifier)])

        text_vector = np.concatenate(
            [qualifier_vector, concept_vector, context_vector], axis=1
        )
        with open(os.path.join(dirname, model_directory, "model_type"), "rb") as f:
            model_type = pickle.load(f)
        with open(os.path.join(dirname, model_directory, "encoder_type"), "rb") as f:
            encoder_type = pickle.load(f)

        pred_type = model_type.predict_proba(text_vector)

        return {
            encoder_type.inverse_transform([0])[0]: pred_type[0][0],
            encoder_type.inverse_transform([1])[0]: pred_type[0][1],
            encoder_type.inverse_transform([2])[0]: pred_type[0][2],
            encoder_type.inverse_transform([3])[0]: pred_type[0][3],
            encoder_type.inverse_transform([4])[0]: pred_type[0][4],
            encoder_type.inverse_transform([5])[0]: pred_type[0][5],
        }

        # return {
        #     'date': pred_type[0][0],
        #     'enumeration': pred_type[0][1],
        #     'integer': pred_type[0][2],
        #     'string': pred_type[0][3],
        #     'time': pred_type[0][4]
        # }


if __name__ == "__main__":
    predictor = ConceptsPrediction()

    # predictor.train_models()

    print("For concept number  ", predictor.predict_category_with_probability("number"))
    print(
        "For concept passport number  ",
        predictor.predict_category_with_probability("passport number", "Person"),
    )
    print(
        "For concept plate  ",
        predictor.predict_category_with_probability("plate", "car"),
    )
    print("For concept city  ", predictor.predict_category_with_probability("city"))
    print("For concept index  ", predictor.predict_category_with_probability("index"))
    print(
        "For concept spots  ",
        predictor.predict_category_with_probability("spots", "bike station"),
    )
    print(
        "For concept code  ",
        predictor.predict_category_with_probability("code", "bike"),
    )
    print(
        "For concept resolution  ",
        predictor.predict_category_with_probability("resolution"),
    )

    # print("For concept number of nights", predictor.predict_category("number of nights"))
    # print("For concept nights", predictor.predict_category("nights"))
    # print("For concept number  ", predictor.predict_category("number"))
    # print("For concept daily rate", predictor.predict_category("daily rate"))
    # print("For concept rate", predictor.predict_category("rate"))
    # print("For concept books", predictor.predict_category("books"))

    # print(
    #     "Prediction for 'presentation time':",
    #     predictor.predict_category("presentation time"),
    # )
    #
    # print("For concept User ", predictor.predict_category("user"))
    # print("For concept Day ", predictor.predict_category("day"))
    # print("\nFor concept name ", predictor.predict_category("name"))
    # print("\nFor concept University ", predictor.predict_category("University"))
    # print("\nFor concept Department ", predictor.predict_category("Department"))
    # print("\nFor concept email ", predictor.predict_category("Email"))
    # print("\nFor concept number ", predictor.predict_category("number"))
    # print("\nFor concept address ", predictor.predict_category("address"))
    # print("\nFor concept ID ", predictor.predict_category("ID"))

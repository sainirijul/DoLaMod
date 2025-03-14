import os
import re

import pandas as pd
from concept_prediction_transformers import ConceptPrediction

# TODO below code is used for preprocessing, you should not need it as test_data is in required format
# import stanza
# Load the English NLP pipeline
# stanza.download('en')  # Only needed once
# nlp = stanza.Pipeline('en')
#
#
# def split_concept(concept):
#     parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)(?![a-z0-9])(?![A-Z0-9])', concept)
#     return " ".join([item.lower() for item in parts])
#
#
# def get_main_noun_in_attribute(attribute):
#     parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)(?![a-z0-9])(?![A-Z0-9])', attribute)
#     if len(parts) == 1:
#         return attribute, ''
#     else:
#         attributes = " ".join([item.lower() for item in parts])
#         result = nlp(attributes)
#         remaining_part = ''
#
#         for sentence in result.sentences:
#             for word in sentence.words:
#                 if word.deprel == 'root':
#                     return word.text, remaining_part
#                 else:
#                     remaining_part += word.text + " "
#
#         return parts[-1], " ".join(parts[0:-1])


def test_attributes_types(version="simple"):
    predictor = ConceptPrediction()
    result = pd.DataFrame(
        columns=[
            "concept",
            "qualifier",
            "context",
            "type",
            "prediction",
            "probabilities",
            "result",
        ]
    )

    test_data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "attribute_datatype_test_data.csv")
    )
    test_data = pd.read_csv(test_data_path)
    for i, row in test_data.iterrows():
        concept = row["concept"]
        context = row["context"]
        qualifier = row["qualifier"]

        if pd.isna(qualifier):
            qualifier = ""
        datatype = row["type"]
        prediction = predictor.predict_category_with_probability(
            concept, context, qualifier
        )

        result.loc[len(result)] = [
            concept,
            qualifier,
            context,
            datatype,
            prediction,
            "",
            "",
        ]

    correct_count = 0
    for _, row in result.iterrows():
        if row["type"] == "Int" and row["prediction"] == "integer":
            correct_count += 1
            row["result"] = "True"
        elif row["type"] == "Enum" and row["prediction"] == "enumeration":
            correct_count += 1
            row["result"] = "True"
        elif row["type"] == "Int" and row["prediction"] == "float":
            correct_count += 1
            row["result"] = "True"
        elif row["type"] == "String" and row["prediction"] == "string":
            correct_count += 1
            row["result"] = "True"
        elif row["type"] == "Double" and (
            row["prediction"] == "integer" or row["prediction"] == "float"
        ):
            correct_count += 1
            row["result"] = "True"
        else:
            row["result"] = "False"

    accuracy = correct_count / len(result)

    print(accuracy)

    result.to_csv(f"results_temp_qual.csv", index=False)


test_attributes_types()

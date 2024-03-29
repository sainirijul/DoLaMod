import spacy
from ml.concepts_prediction import ConceptsPrediction, GloveVectorizer
from rules_nlp.concepts_extractor import ConceptsExtractor
from rules_nlp.relationships_extractor import RelationshipsExtractor


class Dolamod:
    def __init__(self, domain_description: str) -> None:
        self.domain_description = domain_description
        self.language_model = spacy.load("en_core_web_lg")
        self.domain_doc = self.language_model(self.domain_description)
        self.glove = GloveVectorizer(self.language_model)

    """
    Execute both concepts extractor and relationships extractor.
    """

    def process(self):
        concepts_extractor = ConceptsExtractor()
        relationships_extractor = RelationshipsExtractor()
        sentences = [sent.text.strip() for sent in self.domain_doc.sents]
        for sdx, sent in enumerate(sentences):
            sdx = "S" + str(sdx)
            preprocessed_sent = sent.replace(".", "")
            concepts_extractor.extract_candidate_concepts(
                self.language_model(preprocessed_sent), sdx
            )
            relationships_extractor.extract_candidate_relationships(
                concepts_extractor.df_chunks,
                concepts_extractor.df_concepts,
                self.language_model,
                self.language_model(preprocessed_sent),
                sdx,
            )

        print("Concepts Trace Model \n", concepts_extractor.df_concepts, "\n\n")
        print(
            "Relationships Trace Model \n",
            relationships_extractor.df_class_associations,
            "\n\n",
        )
        print("\nPredicted Category/Type")

        base_values = concepts_extractor.df_concepts["lemmatized_text"].values
        concepts_list = []
        for bv in base_values:
            if bv != "NONE":
                concepts_list.append(bv)

        (
            Y_pred_label_cat,
            Y_pred_label_type,
            class_prob_cat,
            class_prob_type,
        ) = ConceptsPrediction().predict_category(concepts_list, self.glove)
        print(Y_pred_label_cat, Y_pred_label_type, class_prob_cat, class_prob_type)


if __name__ == "__main__":
    dolamod = Dolamod(
        "University is composed of departments. Each department has a title."
    )
    dolamod.process()

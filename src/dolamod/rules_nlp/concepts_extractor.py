from typing import Any, List

import pandas as pd


class ConceptsExtractor:
    def __init__(self) -> None:
        self.candidates: List[Any] = []
        self.df_concepts = pd.DataFrame(
            columns=["s_id", "token", "position", "lemmatized_text", "sent"]
        )
        self.df_chunks = pd.DataFrame()

    def filterNounChunks(self, chunk_string):
        if len(chunk_string) > 1:
            list_except_last_ele = [item.tag_ for item in chunk_string[0:-1]]
            new_list = []
            ignore_tags = ["NNS"]
            new_list.append(chunk_string[-1])
            index = len(list_except_last_ele) - 1
            while index >= 0:
                if list_except_last_ele[index] in ignore_tags:
                    break
                else:
                    new_list.append(chunk_string[index])
                index -= 1
            new_list.reverse()
            return new_list
        else:
            return chunk_string

    def find_head(self, child):
        flag = True
        while flag:
            if child.head and child.head != child:
                if child.head.pos_ in ("NOUN", "PROPN") and child.head.i > child.i:
                    base_head = child.head
                    flag = False
                else:
                    base_head = self.find_head(child.head)
                    flag = False
            else:
                base_head = child
                flag = False
        return base_head

    def extract_candidate_concepts(self, doc, sdx):
        sentence_candidates, chunk_string, lemmatized_cadidates = [], [], []
        lemmas, capitalized_lemmas = "", ""
        noun_tags = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP"]
        verb_tags = ["VBN", "VB"]
        comp_deps = ["compound", "conj", "com", "cc", "comp", "amod"]
        tag_for_prn_chunks = ["NNP", "NNPS"]
        no_shape = ["x", "X", "-", ".", "d"]

        for chunks in doc.noun_chunks:
            chunk_string = [chunk for chunk in chunks]
            lemmatized_cadidates = []
            lemm_cadidates = []
            rem = []

            chunk_string = self.filterNounChunks(chunk_string)
            for chunk_ele in chunk_string:
                if (chunk_ele.shape_ not in no_shape) and (
                    (chunk_ele.tag_ in noun_tags)
                    or (
                        (chunk_ele.tag_ in tag_for_prn_chunks)
                        and chunk_ele.dep_ == "nsubj"
                    )
                    or (chunk_ele.tag_ in verb_tags and chunk_ele.dep_ in comp_deps)
                ):
                    if chunk_ele.dep_ in ("nmod", "amod", "conj", "cc", "compound"):
                        lemmas = chunk_ele.lemma_
                        capitalized_lemmas = lemmas.title()
                        capitalized_lemmas = lemmas.title()
                        lemm_cadidates.append("".join(capitalized_lemmas))
                        rem.append(chunk_ele)

                        new_row = {
                            "s_id": sdx,
                            "token": chunk_ele,
                            "position": chunk_ele.i,
                            "lemmatized_text": capitalized_lemmas,
                            "sent": doc.text,
                        }
                        new_row_df = pd.DataFrame(new_row, index=[0])
                        self.df_concepts = pd.concat(
                            [self.df_concepts, new_row_df], ignore_index=True
                        )
                        hyphen_flag = False
                        for cs in chunk_string:
                            if cs.text in [","]:
                                hyphen_flag = True
                                break
                        base_head = self.find_head(chunk_ele)
                        if (
                            base_head.pos_ in ("NOUN", "PROPN")
                            and base_head.dep_ not in ("ROOT")
                            and chunk_ele.lemma_ != base_head.lemma_
                            and not hyphen_flag
                        ):
                            base_lemmas = base_head.lemma_
                            base_capitalized_lemmas = base_lemmas.title()
                            lemm_cadidates.append("".join(base_capitalized_lemmas))

                            new_row = {
                                "s_id": sdx,
                                "token": base_head,
                                "position": base_head.i,
                                "lemmatized_text": base_capitalized_lemmas,
                                "sent": doc.text,
                            }
                            new_row_df = pd.DataFrame(new_row, index=[0])
                            self.df_concepts = pd.concat(
                                [self.df_concepts, new_row_df], ignore_index=True
                            )
                            rem.append(base_head)
                            m = base_head.head
                            while m and m != m.head:
                                if (
                                    m.pos_ in ("NOUN")
                                    and m.dep_ not in ("ROOT")
                                    and m.lemma_ != base_head.lemma_
                                    and base_head.dep_
                                    in ("compound", "amod", "nmod", "cc")
                                ):
                                    base_lemmas = m.lemma_
                                    base_capitalized_lemmas = base_lemmas.title()
                                    lemm_cadidates.append(
                                        "".join(base_capitalized_lemmas)
                                    )
                                    new_row = {
                                        "s_id": sdx,
                                        "token": m,
                                        "position": m.i,
                                        "lemmatized_text": base_capitalized_lemmas,
                                        "sent": doc.text,
                                    }
                                    new_row_df = pd.DataFrame(new_row, index=[0])
                                    self.df_concepts = pd.concat(
                                        [self.df_concepts, new_row_df],
                                        ignore_index=True,
                                    )
                                    rem.append(m)
                                if m.head and m != m.head:
                                    base_head = m
                                    m = m.head

                        sentence_candidates.append(lemm_cadidates)
                        for lc in rem:
                            if lc in chunk_string:
                                chunk_string.remove(lc)
                        lemm_cadidates, rem = [], []
                        continue

                    else:
                        lemmas = chunk_ele.lemma_
                        capitalized_lemmas = lemmas.title()
                        lemmatized_cadidates.append("".join(capitalized_lemmas))
                        new_row = {
                            "s_id": sdx,
                            "token": chunk_ele,
                            "position": chunk_ele.i,
                            "lemmatized_text": capitalized_lemmas,
                            "sent": doc.text,
                        }

                        new_row_df = pd.DataFrame(new_row, index=[0])
                        self.df_concepts = pd.concat(
                            [self.df_concepts, new_row_df], ignore_index=True
                        )

                if lemmatized_cadidates != []:
                    sentence_candidates += lemmatized_cadidates

        self.df_chunks = self.df_concepts

        for _, sc in enumerate(sentence_candidates):
            length = len(sc)
            if length == 1:
                self.candidates.append("".join(sc))
                continue
            elif length > 1:
                self.candidates.append("".join(sc))

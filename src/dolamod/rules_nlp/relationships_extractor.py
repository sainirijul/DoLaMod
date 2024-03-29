import re
from typing import Any, List

import pandas as pd
from word2number import w2n


class RelationshipsExtractor:
    def __init__(self) -> None:
        self.df_class_associations = pd.DataFrame(
            columns=[
                "sdx",
                "source",
                "source_base",
                "source_multi",
                "assoc_type",
                "assoc_label",
                "assoc_label_id",
                "target_multi",
                "target_base",
                "target",
                "desc_probScore",
            ]
        )

    def find_children_2(self, parent):
        noun_tags = ["NN", "NNS", "NNP", "NNPS"]
        deps = ["conj", "attr", "prep", "pobj", "dobj"]
        global children_list
        flag = False
        if parent.children:
            for c in parent.children:
                if not flag:
                    index = parent.i
                if c.dep_ == "appos":
                    self.find_children2(c)
                if c.i > index and c.dep_ in deps and c.tag_ in noun_tags:
                    children_list.append(c)
                    flag = True
                    self.find_children2(c)
        return children_list

    def append_df_row(self, new_row, df):
        new_row_df = pd.DataFrame(new_row, index=[0])
        df = pd.concat([df, new_row_df], ignore_index=True)
        return df

    def find_children_conj(self, parent):
        noun_tags = ["NN", "NNS", "NNP", "NNPS"]
        # deps = ['conj', 'attr', 'prep', 'pobj', 'dobj']
        deps = ["conj", "attr", "cc"]
        global children_list
        flag = False
        if parent.children:
            for c in parent.children:
                if not flag:
                    index = parent.i
                    if c.i > index and c.dep_ in deps and c.tag_ in noun_tags:
                        children_list.append(c)
                        flag = True
                        self.find_children_2(c)
        return children_list

    """
    # Class to identify Target candidate in a given relationship
    """

    def find_target_classes(self, df_chunks, rel, source_class_df, sdx):
        final_target_list = pd.DataFrame(columns=["target_token", "lemma_text"])
        noun_tags = ["NN", "NNS", "NNP", "NNPS"]
        subject_deps = ["nsubj", "nsubjpass"]
        object_deps = ["pobj", "dobj", "attr", "cc", "conj", "compound", "ROOT", "poss"]
        noun_deps = subject_deps + object_deps
        target_class_list = []
        global children_list
        global head_conj_flag
        children_list = []
        source_index_list = []
        concepts_idx = df_chunks["position"]

        noun_tags = ["NN", "NNS", "NNP", "NNPS"]
        verb_tags = ["VBP", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

        for row in range(len(source_class_df)):
            source_index_list.append((source_class_df.loc[row, "source_token"]).i)
        flag = False
        flag_child = False
        for token in rel.children:
            if token.i > rel.i and token.pos_ == "VERB":
                break
            if token.dep_ == "agent":
                for tc in token.children:
                    if (
                        tc.i > rel.i
                        and tc.i in concepts_idx.values
                        and tc.dep_ in object_deps
                        and tc.tag_ in noun_tags
                        and tc.i not in source_index_list
                    ):
                        target_class_list.append(token)
                        if token.children:
                            children = self.find_children2(token)
                            # for c in children:
                            #     if c.i > token.i and c.dep_ in ['pobj', 'dobj']:
                            #         flag = True
                            #         break
                            # if flag:
                            #     break
                            if children:
                                target_class_list = target_class_list + children
                        flag = True
                        break

            if (
                token.i > rel.i
                and token.i in concepts_idx.values
                and token.dep_ in object_deps
                and token.tag_ in noun_tags
                and token.i not in source_index_list
            ):
                children = token.children

                for c in children:
                    if c.i > token.i and c.dep_ in ["prep"] and c.text == "of":
                        flag_child = True
                        flag_child_count = c.i
                        break
                if not flag_child:
                    children_list = []
                    conj_children = self.find_children_conj(token)
                    for c in conj_children:
                        if (
                            c.i < len(c.doc) - 1
                            and c.i + 1 > token.i
                            and c.doc[c.i + 1].dep_ in ["prep"]
                            and c.doc[c.i + 1].text == "of"
                        ):
                            flag_child = True
                            flag_child_count = c.i
                            break

                if not flag_child or (flag_child and rel.pos_ in ["AUX"]):
                    if (
                        token.i + 1 <= len(token.doc) - 1
                        and token.doc[token.i + 1].text != "of"
                    ) or (token.i == len(token.doc)):
                        target_class_list.append(token)
                        if token.children:
                            children = self.find_children2(token)
                            # for c in children:
                            #     if c.i > token.i and c.dep_ in ['prep'] and c.text == "of":
                            #         flag = False
                            #         break
                            # if flag:
                            #     break
                            if children:
                                for c in children:
                                    # leng = len(token.doc)
                                    # text = token.doc[c.i + 1].text
                                    if (
                                        c.i + 1 <= len(token.doc) - 1
                                        and token.doc[c.i + 1].text != "of"
                                    ) or (c.i >= len(token.doc) - 1):
                                        target_class_list.append(c)

                        flag = True

        if not flag:
            # mt_pr_list = []
            for x in rel.doc:
                if x.text in ["categories", "types"]:
                    for x in rel.doc:
                        # mt_pr_list.append(x.text)
                        if x.text in ["-", "i.e."]:
                            mt_pr = x.i - 1
                            while (
                                mt_pr < len(rel.doc) - 1 and rel.doc[mt_pr].tag_ != "."
                            ):
                                mt_pr = mt_pr + 1
                                if (
                                    (
                                        rel.doc[mt_pr].dep_ in object_deps
                                        or rel.doc[mt_pr].tag_ in ["IN"]
                                        or rel.doc[mt_pr].tag_ in noun_tags
                                    )
                                    and rel.doc[mt_pr].i not in source_index_list
                                    and rel.doc[mt_pr].i in concepts_idx.values
                                ):
                                    target_class_list.append(rel.doc[mt_pr])
                                    flag = True
                            break
                    break

        if not flag and rel.tag_ in ["IN", "TO", "RB"]:
            for token in rel.children:
                if (
                    token.i > rel.i
                    and token.i in concepts_idx.values
                    and token.dep_ in object_deps
                    and rel.tag_ in ["IN"]
                    and token.tag_ in noun_tags
                    and token.i not in source_index_list
                ):
                    target_class_list.append(token)
                    flag = True
                children = self.find_children2(rel.head)
                # for c in children:
                #     if c.i > rel.head.i and c.dep_ in ['pobj', 'dobj']:
                #         flag = True
                #         break
                if children and not flag:
                    for child in children:
                        if (
                            child.i > rel.i
                            and child.i not in source_index_list
                            and child.i in concepts_idx.values
                        ):
                            target_class_list.append(child)
                            flag = True
                # head_conj_flag = False
                # children_conj = find_head_conj(rel.head)
                # if children_conj:
                #     target_class_list.append(rel.head)
                #     target_class_list = target_class_list + children_conj
                #     flag = True

        if not flag:
            count = 1
            while (
                count < len(rel.doc) - rel.i
                and not flag
                and rel.nbor(count).tag_ != "."
            ):
                x = rel.nbor(count)
                # if ((flag_child and rel.nbor(count).i > flag_child_count) or (not flag_child)) and (rel.tag_ in verb_tags and rel.nbor(count).i not in source_index_list and rel.nbor(count).i in concepts_idx.values and rel.nbor(count).tag_ in noun_tags):
                flag_child = False
                children = rel.nbor(count).children
                for c in children:
                    if (
                        c.i > rel.nbor(count).i
                        and c.dep_ in ["prep"]
                        and c.text == "of"
                    ):
                        flag_child = True
                        flag_child_count = c.i
                        break
                if not flag_child:
                    children_list = []
                    conj_children = self.find_children_conj(rel.nbor(count))
                    for c in conj_children:
                        if (
                            c.i < len(c.doc) - 1
                            and c.i + 1 > rel.nbor(count).i
                            and c.doc[c.i + 1].dep_ in ["prep"]
                            and c.doc[c.i + 1].text == "of"
                        ):
                            flag_child = True
                            flag_child_count = c.i
                            break
                leng = len(rel.doc) - 1
                tag = rel.nbor(count).tag_
                dep_ = rel.nbor(count).dep_
                if (
                    (
                        rel.nbor(count).i < len(rel.doc) - 1
                        and rel.doc[rel.nbor(count).i + 1].text != "of"
                    )
                    or (rel.nbor(count).i == len(rel.doc) - 1)
                ) and (
                    ((not flag_child) or (flag_child and rel.pos_ in ["AUX"]))
                    and (
                        (rel.tag_ in verb_tags and rel.nbor(count).dep_ not in ["ROOT"])
                        or (rel.dep_ in ["ROOT"])
                    )
                    and rel.nbor(count).i not in source_index_list
                    and rel.nbor(count).i in concepts_idx.values
                    and rel.nbor(count).tag_ in noun_tags
                ):
                    # if ((rel.nbor(count).i < len(rel.doc)-1 and rel.doc[rel.nbor(count).i + 1].text != "of") or (rel.nbor(count).i == len(rel.doc)-1)) and (rel.tag_ in verb_tags and rel.nbor(count).i not in source_index_list and rel.nbor(count).i in concepts_idx.values and rel.nbor(count).tag_ in noun_tags):

                    # if ((rel.tag_ in verb_tags and rel.nbor(count).dep_ in noun_deps) or rel.tag_ in prep_tags) and (rel.nbor(count).tag_ in noun_tags) and rel.nbor(count).i not in source_index_list:
                    children_a = rel.nbor(count).subtree
                    # for c in children_a:
                    #     if c.i > rel.nbor(count).i and c.dep_ in ['pobj', 'dobj']:
                    #         flag = True
                    #         break
                    # if flag:
                    #     break
                    if (
                        rel.nbor(count).i + 1 <= len(rel.doc) - 1
                        and rel.doc[rel.nbor(count).i + 1].text != "of"
                    ) or (rel.nbor(count).i >= len(rel.doc) - 1):
                        target_class_list.append(rel.nbor(count))
                        if rel.nbor(count).children:
                            children = self.find_children2(rel.nbor(count))
                            if children:
                                target_class_list = target_class_list + children
                            elif (
                                rel.nbor(count).head.tag_ in noun_tags
                                and rel.nbor(count).head.i in concepts_idx.values
                                and rel.nbor(count).head.i not in source_index_list
                            ):
                                children = self.find_children2(rel.nbor(count).head)
                                if children:
                                    # target_class_list.append(rel.nbor(count).head)
                                    for c in children:
                                        if (
                                            c.i + 1 <= len(rel.doc) - 1
                                            and token.doc[c.i + 1].text != "of"
                                        ) or (c.i == len(token.doc) - 1):
                                            target_class_list.append(c)
                                        # elif c.i == len(rel.doc):
                                        #     target_class_list.append(c)
                                        # target_class_list = target_class_list + children

                        flag = True
                count = count + 1

        if flag and target_class_list != []:
            for target_left in target_class_list:
                for row in range(len(df_chunks)):
                    # if target_left.i == df_chunks.loc[row, 'position']:
                    if (
                        sdx == df_chunks.loc[row, "s_id"]
                        and target_left.i == (df_chunks.loc[row, "token"]).i
                    ):
                        new_row = {
                            "target_token": target_left,
                            "lemma_text": df_chunks.loc[row, "lemmatized_text"],
                        }
                        new_row_df = pd.DataFrame(new_row, index=[0])
                        final_target_list = pd.concat(
                            [final_target_list, new_row_df], ignore_index=True
                        )

        return final_target_list

    """
    # To identify relationship type
    """

    def find_association_similarity(self, rel, nlp, docs):
        df_desc_rel = pd.DataFrame(columns=["assoc_type", "prob_score"])
        found_rel = ""
        ags_scores, gs_scores, att_scores, asso_scores = [], [], [], []
        for c in docs:
            if c.text in ["of types", "categories", "groups"]:
                rel = "types"
                break
        relation = nlp(rel)

        aggregation_samples = [
            "contain",
            "known",
            "is made up of",
            "includes",
            "constitute",
            "compose",
            "comprise",
            "that includes",
        ]
        attribute_samples = [
            "of",
            "records",
            "defines",
            "has",
            "have",
            "identified",
            "characterized",
            "property",
            "of",
            "characterized",
            "attributed",
        ]
        generalization_samples = [
            "be",
            "can be",
            "is a",
            "categorizes",
            "types of",
            "type of",
            "kind of",
            "may be",
            "be",
            "maybe",
            "such as",
            "for example",
        ]
        association_samples = [
            "are",
            "is",
            "may",
            "write",
            "associated",
            "assigned",
            "appointed",
            "related",
            "linked",
            "corresponding",
            "accompanying",
            "work",
        ]
        # aggregation_tokens = nlp(aggregation_samples)
        # generalization_tokens = nlp(generalization_samples)
        # attribute_tokens = nlp(attribute_samples)
        for ags in aggregation_samples:
            ags_scores.append((relation).similarity(nlp(ags)))

        for gs in generalization_samples:
            gs_scores.append((relation).similarity(nlp(gs)))

        for ats in attribute_samples:
            att_scores.append((relation).similarity(nlp(ats)))

        for asso in association_samples:
            asso_scores.append((relation).similarity(nlp(asso)))

        max_ags_scores = max(ags_scores)
        max_gs_scores = max(gs_scores)
        max_att_scores = max(att_scores)
        max_asso_scores = max(asso_scores)

        df_desc_rel = self.append_df_row(
            {"assoc_type": "association", "prob_score": max_asso_scores}, df_desc_rel
        )
        df_desc_rel = self.append_df_row(
            {"assoc_type": "aggregation", "prob_score": max_ags_scores}, df_desc_rel
        )
        df_desc_rel = self.append_df_row(
            {"assoc_type": "attributes", "prob_score": max_att_scores}, df_desc_rel
        )
        df_desc_rel = self.append_df_row(
            {"assoc_type": "generalization", "prob_score": max_gs_scores}, df_desc_rel
        )

        if df_desc_rel.prob_score.max() < 0.45:
            found_rel = "association"
        else:
            found_rel = df_desc_rel.loc[
                df_desc_rel.prob_score == df_desc_rel.prob_score.max(), "assoc_type"
            ].values[0]
        return found_rel, df_desc_rel

    def find_children2(self, parent):
        noun_tags = ["NN", "NNS", "NNP", "NNPS"]
        deps = ["conj", "attr", "prep", "pobj", "dobj", "appos"]
        global children_list
        flag = False
        if parent.children:
            for c in parent.children:
                if not flag:
                    index = parent.i
                if c.dep_ == "appos":
                    self.find_children2(c)
                if c.i > index and c.dep_ in deps and c.tag_ in noun_tags:
                    if c.i < len(c.doc) - 1:
                        if c.doc[c.i + 1].text != "of":
                            children_list.append(c)
                            flag = True
                            self.find_children2(c)
                        elif c.doc[c.i + 1].text == "of":
                            self.find_children2(c.doc[c.i + 1])
                    else:
                        children_list.append(c)
                        flag = True
                        self.find_children2(c)
        return children_list

    """
    # To identify conjugate source candidate in a given relationship
    """

    def find_children_conj_src(self, parent):
        noun_tags = ["NN", "NNS", "NNP", "NNPS"]
        deps = ["conj", "attr", "prep", "pobj", "dobj"]
        # deps = ['conj', 'attr', 'cc']
        global children_list
        flag = False
        if parent.children:
            for c in parent.children:
                if not flag:
                    index = parent.i
                if c.i > index and c.dep_ in deps and c.tag_ in noun_tags:
                    children_list.append(c)
                    flag = True
                    self.find_children_2(c)
        return children_list

    """
    # To identify source candidate in a given relationship
    """

    def find_source_class(self, df_chunks, rel, sdx):
        final_source_list = pd.DataFrame(
            columns=["source_token", "lemma_text", "relation"]
        )
        subject_deps = ["nsubj", "nsubjpass"]
        object_deps = ["pobj", "dobj", "attr"]
        noun_tags = ["NN", "NNS", "NNP", "NNPS"]
        noun_deps = subject_deps + object_deps
        source_class_list = []
        global children_list
        children_list = []
        flag = False
        verb_tags = ["VBP", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        concepts_idx = df_chunks["position"]

        for token in rel.children:
            if token.i < rel.i and token.pos_ == "VERB":
                continue
            if (
                token.i < rel.i
                and token.i in concepts_idx.values
                and token.dep_ in subject_deps
                and token.tag_ in noun_tags
            ):
                source_class_list.append(token)
                if token.children:
                    children = self.find_children2(token)
                    if children:
                        source_class_list = source_class_list + children
                conj_children = self.find_children_conj_src(token)
                if conj_children:
                    source_class_list = source_class_list + conj_children
                flag = True

        if not flag:
            # if rel.head.tag_ in noun_tags and rel.head.dep_ in noun_deps:
            if rel.head.tag_ in noun_tags and rel.head.i in concepts_idx.values:
                source_class_list.append(rel.head)
                if rel.head.children:
                    children = self.find_children2(rel.head)
                    if children:
                        source_class_list = source_class_list + children
                conj_children = self.find_children_conj_src(rel.head)
                if conj_children:
                    source_class_list = source_class_list + conj_children
                flag = True

        if (
            len(rel.doc) > rel.i - 1
            and len(rel.doc) > rel.i + 1
            and len(rel.doc) > rel.i - 2
            and len(rel.doc) > rel.i + 2
        ):
            if (
                not flag
                and (
                    rel.doc[rel.i - 1].tag_ not in verb_tags
                    and rel.doc[rel.i + 1].tag_ not in verb_tags
                )
                and (
                    rel.doc[rel.i - 2].tag_ not in verb_tags
                    and rel.doc[rel.i + 2].tag_ not in verb_tags
                )
                and rel.doc[rel.i - 1].tag_ in ["TO", "IN"]
            ):
                # if (not flag and (rel.doc[rel.i - 1].tag_ in ['TO','IN'] or rel.doc[rel.i +])):
                m = rel.i - 1
                while m > 0 and not flag and rel.doc[m].tag_ != ".":
                    if (
                        rel.doc[m].dep_ in noun_deps
                        and rel.doc[m].i in concepts_idx.values
                    ):
                        source_class_list.append(rel.doc[m])
                        if rel.doc[m].children:
                            children = self.find_children2(rel.doc[m])
                            if children:
                                source_class_list = source_class_list + children
                        flag = True
                        conj_children = self.find_children_conj(rel.head)
                        if conj_children:
                            source_class_list = source_class_list + conj_children
                    m = m - 1

        if flag and source_class_list != []:
            for source_left in source_class_list:
                for row in range(len(df_chunks)):
                    # if source_left.i == df_chunks.loc[row, 'position']:
                    abc = df_chunks.loc[row, "token"]

                    if (
                        sdx == df_chunks.loc[row, "s_id"]
                        and source_left.i == (df_chunks.loc[row, "token"]).i
                    ):
                        new_row = {
                            "source_token": source_left,
                            "lemma_text": df_chunks.loc[row, "lemmatized_text"],
                            "relation": rel,
                        }
                        new_row_df = pd.DataFrame(new_row, index=[0])
                        final_source_list = pd.concat(
                            [final_source_list, new_row_df], ignore_index=True
                        )

        return final_source_list

    def find_multiplicity(self, nlp, class_token, token_lemma):
        # plural_nouns = ['NNPS', 'NNS']
        noun_tags = ["NNP", "NN", "NNPS", "NNS"]
        id_tags = ["CD", "JJ", "DT", "JJR", "JJS"]
        many_score, one_score = pd.DataFrame(
            columns=["id", "idx", "score"]
        ), pd.DataFrame(columns=["id", "score"])
        many_score2, one_score2 = pd.DataFrame(
            columns=["id", "idx", "score"]
        ), pd.DataFrame(columns=["id", "score"])
        samples_one = [
            "one",
            "an",
            "a",
            "the",
            "single",
            "only",
            "unique",
            "exclusive",
            "particular",
            "individual",
            "lone",
            "only",
            "solo",
            "lone",
            "separate",
        ]
        samples_multi = [
            "many",
            "some",
            "multiple",
            "diferent",
            "diverse",
            "certain",
            "several",
            "various",
            "few",
            "number",
            "huge",
            "numbers",
            "large",
            "much",
            "more",
        ]

        # multi_tokens = nlp(samples_multi)
        # one_tokens = nlp(samples_one)

        chunks = re.findall("[A-Z][^A-Z]*", token_lemma)
        if len(chunks) > 1 and (nlp(chunks[0]))[0].tag_ in id_tags:
            for sm in samples_multi:
                new_row = {
                    {"id": chunks[0], "score": (nlp(chunks[0])).similarity(nlp(sm))},
                }
                new_row_df = pd.DataFrame(new_row, index=[0])
                many_score = pd.concat([many_score, new_row_df], ignore_index=True)

            for so in samples_one:
                new_row = {
                    {"id": chunks[0], "score": (nlp(chunks[0])).similarity(nlp(so))},
                }
                new_row_df = pd.DataFrame(new_row, index=[0])
                one_score = pd.concat([one_score, new_row_df], ignore_index=True)

            # for val in many_score:
            #     if val != None :
            #         many_score2.append(val)

            # for val in one_score:
            #     if val != None :
            #         one_score2.append(val)
            # abc = max(many_score)
            # deff = max(one_score)

            many_score.mask(many_score.eq("None")).dropna()
            one_score.mask(one_score.eq("None")).dropna()

            if (
                many_score.score.max() > one_score.score.max()
                and many_score.score.max() > 0.8
                and chunks[0]
            ):
                found_id = many_score.loc[
                    many_score.score == many_score.score.max(), "id"
                ].values[0]
                return "0..*", found_id, "", many_score.score.max

            # if (max(many_score) > max(one_score)) and max(many_score) > 0.8 and chunks[0]:
            #     return '0..*'

        identifier = class_token
        count = -1
        while count >= (-(class_token.i)):
            # if class_token.nbor(count).pos_ == 'DET' or class_token.nbor(count).dep_ == 'nummod':
            token_text = class_token.nbor(count).text
            token_text = token_text.title()
            if token_text not in chunks and class_token.nbor(count).tag_ in noun_tags:
                return "?", "", "", ""
            if token_text not in chunks and (
                class_token.nbor(count).tag_ in id_tags
                or class_token.nbor(count).dep_ == "nummod"
            ):
                identifier = class_token.nbor(count)
                if (
                    identifier
                    and identifier != class_token
                    and identifier.tag_ in id_tags
                    and identifier.dep_ != "nummod"
                ):
                    identifier_index = identifier.i
                    identifier_token = nlp(identifier.text)

                    for sm in samples_multi:
                        new_row = {
                            "id": identifier_token,
                            "idx": identifier_index,
                            "score": (identifier_token).similarity(nlp(sm)),
                        }

                        new_row_df = pd.DataFrame(new_row, index=[0])
                        many_score2 = pd.concat(
                            [many_score2, new_row_df], ignore_index=True
                        )

                    for so in samples_one:
                        # one_score.append((identifier_token).similarity(nlp(so)))
                        new_row = {
                            "id": identifier_token,
                            "idx": identifier_index,
                            "score": (identifier_token).similarity(nlp(so)),
                        }
                        new_row_df = pd.DataFrame(new_row, index=[0])
                        one_score2 = pd.concat(
                            [one_score2, new_row_df], ignore_index=True
                        )

                    # for val in many_score:
                    #     if val != None:
                    #         many_score2.append(val)

                    # for val in one_score:
                    #     if val != None:
                    #         one_score2.append(val)

                    many_score2.mask(many_score.eq("None")).dropna()
                    one_score2.mask(one_score.eq("None")).dropna()

                    # if many_score.score.max() > one_score.score.max() and many_score.score.max() > 0.8 and chunks[0]:
                    #     found_id = many_score.loc[many_score.score == many_score.score.max(),'id'].values[0]
                    #     return '0..*',found_id,many_score.score.max

                    if len(one_score2) != 0 and len(many_score2) != 0:
                        max_many_sco = many_score2.score.max()
                        max_one_sco = one_score2.score.max()

                        if (max_many_sco >= max_one_sco) and max_many_sco > 0.7:
                            found_id = many_score2.loc[
                                many_score2.score == many_score2.score.max(), "id"
                            ].values[0]
                            found_idx = many_score2.loc[
                                many_score2.score == many_score2.score.max(), "idx"
                            ].values[0]
                            return "0..*", found_id, found_idx, max_many_sco
                        elif (max_one_sco >= max_many_sco) and max_one_sco > 0.7:
                            found_id = one_score2.loc[
                                one_score2.score == one_score2.score.max(), "id"
                            ].values[0]
                            found_idx = one_score2.loc[
                                one_score2.score == one_score2.score.max(), "idx"
                            ].values[0]
                            return "0..1", found_id, found_idx, max_one_sco
                elif identifier and identifier.dep_ == "nummod":
                    if identifier.shape_ == "d":
                        concan_value = "0.." + str(identifier.text) + "'"
                    else:
                        value = w2n.word_to_num(identifier.text)
                        concan_value = "0.." + str(value) + "'"
                    if concan_value:
                        return concan_value, identifier.text, identifier.i, ""
            count = count - 1
        # elif (identifier != class_token and identifier.tag_ in singular_nouns) or class_token.tag_ in singular_nouns:
        #     return "0..1"
        # elif (identifier != class_token and identifier.tag_ in plural_nouns) or class_token.tag_ in plural_nouns:
        #     return "0..*"
        else:
            return "?", "", "", ""

    def intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value not in lst2]
        return lst3

    def extract_candidate_relationships(self, df_chunks, candidates, nlp, doc, sdx):
        verb_tags = ["VBP", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        prep_tags = ["IN", "TO"]
        rel_tags = verb_tags

        # Check for all Verb forms
        # relation_tokens = [token for token in doc if (token.tag_ in rel_tags) or token.dep_ in ["ROOT"]]
        relation_tokens = [
            token
            for token in doc
            if (token.tag_ in rel_tags)
            or (token.tag_ in ["IN"] and token.dep_ in ["prep"])
        ]
        # for rel_token in relation_tokens:
        for rel_token in relation_tokens:
            # Find source concepts
            verb_index = rel_token.i

            # if rel_token.tag_ in verb_tags or (rel_token.tag_ in prep_tags and doc[verb_index-1].tag_ not in verb_tags and doc[verb_index+1].tag_ not in verb_tags):
            if (rel_token.tag_ in verb_tags) or (rel_token.dep_ in ["ROOT"]):
                source_concepts_df = self.find_source_class(df_chunks, rel_token, sdx)
                if source_concepts_df.empty:
                    continue
                else:
                    # Find target concepts
                    target_classes_df = self.find_target_classes(
                        df_chunks, rel_token, source_concepts_df, sdx
                    )
                    if not target_classes_df.empty:
                        (
                            association_type,
                            df_desc_rel,
                        ) = self.find_association_similarity(
                            rel_token.text, nlp, rel_token.doc
                        )

                        df_desc_rel.sort_values(by="prob_score", ascending=False)
                        # for row in range(len(df_desc_rel)):
                        #     gkb_descriptive_relationship = gkb_descriptive_relationship.append(
                        #         {'rel_id':rel_id,'sdx':sdx,'relationship': df_desc_rel.loc[row,'assoc_type'],'prob_score':df_desc_rel.loc[row,'prob_score']}, ignore_index=True)

                        for row1 in range(len(source_concepts_df)):
                            (
                                source_multi,
                                id_src_mul,
                                src_id_mul_i,
                                score_src_mul,
                            ) = self.find_multiplicity(
                                nlp,
                                source_concepts_df.loc[row1, "source_token"],
                                source_concepts_df.loc[row1, "lemma_text"],
                            )
                            # if rem_word_source != 'zombie':
                            #     source_class_lemma = re.sub(rem_word_source.title(), '', source_class_lemma)
                            for row2 in range(len(target_classes_df)):
                                # target_token = target_classes_df.loc[row2, 'target_token']
                                # target_lemma = target_classes_df.loc[row2, 'lemma_text']
                                # association_type, df_desc_rel = find_association_similarity(
                                #     rel_token.text)
                                # if target_lemma:
                                (
                                    target_multi,
                                    id_tar_mul,
                                    tar_id_mul_i,
                                    score_tar_mul,
                                ) = self.find_multiplicity(
                                    nlp,
                                    target_classes_df.loc[row2, "target_token"],
                                    target_classes_df.loc[row2, "lemma_text"],
                                )
                                # if rem_word_target != 'zombie':
                                #     target_lemma = re.sub(
                                #         rem_word_target.title(), '', target_lemma)
                                # if association_type == 'generalization' or association_type == 'attributes':
                                # if association_type == 'generalization':
                                #     source_multi = '-'
                                #     target_multi = '-'

                                new_row = {
                                    "sdx": sdx,
                                    "source": source_concepts_df.loc[
                                        row1, "lemma_text"
                                    ],
                                    "source_base": source_concepts_df.loc[
                                        row1, "source_token"
                                    ],
                                    "source_multi": source_multi,
                                    "src_multi_md": src_id_mul_i,
                                    "assoc_type": association_type,
                                    "assoc_label": rel_token,
                                    "assoc_label_id": rel_token.i,
                                    "target_multi": target_multi,
                                    "tar_multi_md": tar_id_mul_i,
                                    "target_base": target_classes_df.loc[
                                        row2, "target_token"
                                    ],
                                    "target": target_classes_df.loc[row2, "lemma_text"],
                                    "desc_probScore": pd.Series(
                                        df_desc_rel["prob_score"]
                                    ),
                                }

                                self.df_class_associations = self.append_df_row(
                                    new_row, self.df_class_associations
                                )

                                # gkb_descriptive_configuration = gkb_descriptive_configuration.append(
                                #     {'rel_id': rel_id,'sdx':sdx,'relationshipToken':rel_token,'relationshipLabel': rel_token.text,'src':source_concepts_df.loc[row1,'source_token'],'src_lemma': source_concepts_df.loc[row1,'lemma_text'],'tar':target_classes_df.loc[row2, 'target_token'],'tar_lemma':target_classes_df.loc[row2, 'lemma_text']}, ignore_index=True)

                                # gkb_descriptive_cardinality = gkb_descriptive_cardinality.append(
                                # {'rel_id':rel_id,'sdx':sdx,'src_cardinality':'zero-to-one','tar_cardinality':'zero-to-many'}, ignore_index=True)

                    else:
                        continue

            else:
                continue

        # print("\n")
        # print("----------------------------------------------------------------------------------------")
        # print("\n")
        # print(self.df_class_associations)
        # print("\n")

        source_concepts = self.df_class_associations["source"]
        target_concepts = self.df_class_associations["target"]

        concepts = []
        for sc in source_concepts:
            concepts.append(sc)
        for tc in target_concepts:
            concepts.append(tc)

        # for c in concepts:
        #     for row in range(len(df_chunks)):
        #         if df_chunks.loc[row, 'lemmatized_text'] == c:
        #             index_list.append(df_chunks.loc[row, 'position'])

        # concepts = set(concepts)

        rem_conc = self.intersection(candidates, concepts)
        # print("Remaining Concepts after First Iteration", rem_conc)
        if rem_conc != []:
            index_list = []

            for rc in rem_conc:
                # for rc in concepts:
                for row in range(len(df_chunks)):
                    if df_chunks.loc[row, "lemmatized_text"] == rc:
                        index_list.append(df_chunks.loc[row, "position"])

            for idx in index_list:
                token_flag = False
                doc_token = ""
                if (
                    len(doc) > idx + 1
                    and idx - 1 >= 0
                    and (
                        doc[idx + 1].tag_ in prep_tags
                        or (doc[idx + 1].tag_ == "RB" and doc[idx + 1].shape_ == "x.x.")
                    )
                ):
                    doc_token = doc[idx + 1]
                    token_flag = True
                elif (
                    idx - 1 >= 0
                    and len(doc) > idx + 1
                    and (
                        doc[idx - 1].tag_ in prep_tags
                        or (doc[idx - 1].tag_ == "RB" and doc[idx - 1].shape_ == "x.x.")
                    )
                ):
                    doc_token = doc[idx - 1]
                    token_flag = True
                elif (
                    len(doc) > idx + 2
                    and idx - 2 >= 0
                    and doc[idx + 1].dep_ in ["punct"]
                    and (
                        doc[idx + 2].tag_ in prep_tags
                        or (doc[idx + 2].tag_ == "RB" and doc[idx + 2].shape_ == "x.x.")
                    )
                ):
                    doc_token = doc[idx + 2]
                    token_flag = True
                elif (
                    len(doc) > idx + 2
                    and idx - 2 >= 0
                    and doc[idx - 1].dep_ in ["punct"]
                    and (
                        doc[idx - 2].tag_ in prep_tags
                        or (doc[idx - 2].tag_ == "RB" and doc[idx - 2].shape_ == "x.x.")
                    )
                ):
                    doc_token = doc[idx - 2]
                    token_flag = True

                if token_flag:
                    source_concepts_df = self.find_source_class2(doc_token, sdx)
                    if source_concepts_df.empty:
                        continue
                    else:
                        # Find target concepts
                        target_classes_df = self.find_target_classes(
                            df_chunks, doc_token, source_concepts_df, sdx
                        )
                        if not target_classes_df.empty:
                            (
                                association_type,
                                df_desc_rel,
                            ) = self.find_association_similarity(
                                rel_token.text, nlp, rel_token.doc
                            )

                            df_desc_rel.sort_values(by="prob_score", ascending=False)
                            # for row in range(len(df_desc_rel)):
                            #     gkb_descriptive_relationship = gkb_descriptive_relationship.append(
                            #         {'rel_id':rel_id,'sdx':sdx,'relationship': df_desc_rel.loc[row,'assoc_type'],'prob_score':df_desc_rel.loc[row,'prob_score']}, ignore_index=True)

                            for row1 in range(len(source_concepts_df)):
                                (
                                    source_multi,
                                    id_src_mul,
                                    src_id_mul_i,
                                    score_src_mul,
                                ) = self.find_multiplicity(
                                    nlp,
                                    source_concepts_df.loc[row1, "source_token"],
                                    source_concepts_df.loc[row1, "lemma_text"],
                                )
                                # if rem_word_source != 'zombie':
                                #     source_class_lemma = re.sub(rem_word_source.title(), '', source_class_lemma)
                                for row2 in range(len(target_classes_df)):
                                    # target_token = target_classes_df.loc[row2, 'target_token']
                                    # target_lemma = target_classes_df.loc[row2, 'lemma_text']
                                    # association_type, df_desc_rel = find_association_similarity(
                                    #     doc_token.text)
                                    (
                                        target_multi,
                                        id_tar_mul,
                                        tar_id_mul_i,
                                        score_tar_mul,
                                    ) = self.find_multiplicity(
                                        nlp,
                                        target_classes_df.loc[row2, "target_token"],
                                        target_classes_df.loc[row2, "lemma_text"],
                                    )
                                    # if rem_word_target != 'zombie':
                                    #     target_lemma = re.sub(
                                    #         rem_word_target.title(), '', target_lemma)
                                    # if association_type == 'generalization' or association_type == 'attributes':
                                    #     source_multi = '-'
                                    #     target_multi = '-'

                                    new_row = {
                                        "sdx": sdx,
                                        "source": source_concepts_df.loc[
                                            row1, "lemma_text"
                                        ],
                                        "source_base": source_concepts_df.loc[
                                            row1, "source_token"
                                        ],
                                        "source_multi": source_multi,
                                        "src_multi_md": src_id_mul_i,
                                        "assoc_type": association_type,
                                        "assoc_label": doc_token,
                                        "assoc_label_id": doc_token.i,
                                        "target_multi": target_multi,
                                        "tar_multi_md": tar_id_mul_i,
                                        "target_base": target_classes_df.loc[
                                            row2, "target_token"
                                        ],
                                        "target": target_classes_df.loc[
                                            row2, "lemma_text"
                                        ],
                                        "desc_probScore": list(
                                            df_desc_rel["prob_score"]
                                        ),
                                    }
                                    new_row_df = pd.DataFrame(new_row, index=[0])
                                    self.df_class_associations = pd.concat(
                                        [self.df_class_associations, new_row_df],
                                        ignore_index=True,
                                    )

                                #     gkb_descriptive_configuration = gkb_descriptive_configuration.append(
                                #    {'rel_id': rel_id,'sdx':sdx,'relationshipToken':rel_token,'relationshipLabel': rel_token.text,'src':source_concepts_df.loc[row1,'source_token'],'src_lemma': source_concepts_df.loc[row1,'lemma_text'],'tar':target_classes_df.loc[row2, 'target_token'],'tar_lemma':target_classes_df.loc[row2, 'lemma_text']}, ignore_index=True)

                                #     gkb_descriptive_cardinality = gkb_descriptive_cardinality.append(
                                # { 'rel_id':rel_id,'sdx':sdx,'src_cardinality':'zero-to-one','tar_cardinality':'zero-to-many'}, ignore_index=True)

                        else:
                            continue
                else:
                    continue

            source_concepts = self.df_class_associations["source"]
            target_concepts = self.df_class_associations["target"]

            concepts = []
            for sc in source_concepts:
                concepts.append(sc)
            for tc in target_concepts:
                concepts.append(tc)

            concepts = set(concepts)

            rem_conc = self.intersection(candidates, concepts)
        # print("Remaining Concepts after second Iteration", rem_conc)

        self.df_class_associations = self.df_class_associations.drop_duplicates(
            subset=["source", "source_multi", "target_multi", "target"], keep="last"
        )

import http.server
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
from io import BytesIO
#from pprint import pprint

import pandas as pd
import spacy
from spacy_langdetect import LanguageDetector
import gzip
import re

import requests
import json

import tensorflow as tf
import numpy as np
from numpy.linalg import norm
import time
import os

from googletrans import Translator

from translation_matrix import TranslationMatrix
from translation_matrix import Space

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def switch_language(language):
    if language == 'en':
        language = 'de'
    else:
        language = 'en'
    return language

def language_resolution(message, language):
    if language == 'en':
        nlp_m = nlp_en
    else:
        nlp_m = nlp_de

    return nlp_m(message)

def sentence_vektor(doc):
    sv = 0
    for token in doc:
            sv += token.vector
    return sv / len(doc)

def cos_sim(sv_1, sv_2):
    return np.dot(sv_1, sv_2)/(norm(sv_1)*norm(sv_2))

question_entity_map = {
    'en':   [["what", "object"],
    ["which", "object"],
    ["who", "person"],
    ["where", "place"],
    ["when", "time"]],
    'de':   [["was", "object"],
    ["welche", "object"],
    ["wer", "person"],
    ["wo", "place"],
    ["wann", "time"]]
}

entity_map = {
    "object": [],
    "person": [],
    "place": ["area", "place", "spot", "room", "floor", "side"],
    "time": []
}

language_map = ['en', 'de']

class Objects:
    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]


class Object:
    def __init__(self, name):
        self['name'] = name

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __del__(self):
        pass


class parts_of_the_sentence:
    def __init__(self):
        self.dobj = []
        self.pobj = []
        self.subj = []
        self.amod = []
        self.num = []

class parts_of_the_speech:
    def __init__(self):
        self.nouns = []
        self.verbs = []
        self.wh = []


def question_resolution(mes, language, faq):
    #TODO Sentences with more than 1 question
    if faq:
        q_type = question_entity_map[language][0][1]
    else:
        q_type = None

    sentence_parts = parts_of_the_sentence()
    speech_parts = parts_of_the_speech()
    sentence_parts, speech_parts, root = sentence_processing(mes, sentence_parts, speech_parts)

    if mes[0].pos_ == "VERB" or mes[0].pos_ == "AUX":
        return "yes/no"

    for question in speech_parts.wh:
        for q_e in question_entity_map[language]:
            if mes[0].text == question.text:
                if question.lemma_ == q_e[0]:
                    return q_e[1]

    return q_type
    ###why


def name_resolution(mes, language, q_entity, sentence_parts, speech_parts, root):
    if q_entity == question_entity_map[language][0][1] or q_entity == question_entity_map[language][1][1]:
        if speech_parts.wh:
            for noun in speech_parts.nouns:
                for question in speech_parts.wh:
                    if noun.lemma_ == question.head.lemma_:#TODO noun.text.lower() == question.head.text.lower():
                        return noun.text
        else:
            if sentence_parts.subj:
                for subj in sentence_parts.subj:
                    for noun in speech_parts.nouns:
                        if subj.text == noun.text:
                            return subj.text
            if sentence_parts.pobj:
                return sentence_parts.pobj[0].text

            #print("Search for nouns near the root")
            for noun in speech_parts.nouns:
                if noun.dep_ == "ROOT":
                    return noun.text
                if noun.head.text == root:
                    return noun.text

            # If words are staying together :/
            for i in range(len(mes) - 1):
                if mes[i].text == root:
                    for noun in speech_parts.nouns:
                        if mes[i + 1].text == noun.text:
                            return noun.text
        return q_entity

    elif q_entity == question_entity_map[language][3][1]:
        if sentence_parts.subj:
            if sentence_parts.subj:
                for subj in sentence_parts.subj:
                    for noun in speech_parts.nouns:
                        if subj.text == noun.text:
                            return subj.text
        else:
            #print("Search for nouns near the root")
            for noun in speech_parts.nouns:
                if noun.dep_ == "ROOT":
                    return noun.text
                if noun.head.text == root:
                    return noun.text

            # If words are staying together :/
            for i in range(len(mes) - 1):
                if mes[i].text == root:
                    for noun in speech_parts.nouns:
                        if mes[i + 1].text == noun.text:
                            return noun.text
        return q_entity


    elif q_entity == question_entity_map[language][4][1]:
        pass

    elif q_entity == "yes/no":#"Yes or no?" - this question will not be classified as yes/no, that's choice from 2
        if sentence_parts.subj:
            if sentence_parts.subj:
                for subj in sentence_parts.subj:
                    for noun in speech_parts.nouns:
                        if subj.text == noun.text:
                            return subj.text
        if sentence_parts.pobj:
            return sentence_parts.pobj[0].text
        else:
            #print("Search for nouns near the root")
            for noun in speech_parts.nouns:
                if noun.head.text == root:
                    return noun.text

            # If words are staying together :/
            for i in range(len(mes) - 1):
                if mes[i].text == root:
                    for noun in speech_parts.nouns:
                        if mes[i + 1].text == noun.text:
                            return noun.text

    return "none"######################################################################################################


def sentence_processing(doc, sentence_parts, speech_parts):
    root = ''
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)

    for token in doc:
        #print(token.text, token.tag_, token.head.text, token.dep_, token.lemma_, token.pos_)
        if token.dep_ == "dobj":
            sentence_parts.dobj.append(token)

        elif token.dep_ == "pobj" or token.dep_ == "nk": #nk and sb are german tags
            sentence_parts.pobj.append(token)

        elif token.dep_ == "nsubj" or token.dep_ == "sb":
            sentence_parts.subj.append(token)

        elif token.pos_ == "ADJ":
            sentence_parts.amod.append(token)

        elif token.dep_ == "npadvmod" or token.dep_ == "nummod":
            sentence_parts.num.append(token)

        elif token.dep_ == "ROOT":# and "VB" in token.tag_:
            root = token.text

        if token.tag_.find("NN") != -1 or token.tag_ == "NE":
            speech_parts.nouns.append(token)

        elif "VB" in token.tag_:#or token.tag_ == "MD"
            speech_parts.verbs.append(token)

        elif "W" in token.tag_:
            speech_parts.wh.append(token)

    #print(sentence_parts.dobj, sentence_parts.pobj, sentence_parts.subj, sentence_parts.amod)
    #print(speech_parts.nouns, speech_parts.verbs, root, speech_parts.wh)
    return(sentence_parts, speech_parts, root)


def object_resolution(name, nlp, doc, word_model):
    object = ''
    sentence_parts = parts_of_the_sentence()
    speech_parts = parts_of_the_speech()
    sentence_parts, speech_parts, root = sentence_processing(doc, sentence_parts, speech_parts)
    print(sentence_parts.dobj, sentence_parts.pobj, sentence_parts.subj, sentence_parts.amod)
    print(speech_parts.nouns, speech_parts.verbs, root, speech_parts.wh)
    print("Entities:", [(X.text, X.label_) for X in doc.ents])

    word = nlp(name)
    for noun in speech_parts.nouns:
        for token in word:
            if token.text.lower() in noun.text.lower():
                object += noun.text.lower() + ' '
    object = object[:-1]
    """
    for noun in speech_parts.nouns:
        if levenshtein_ratio_and_distance(noun.text, name, ratio_calc=True) >= 0.8:
            print(levenshtein_ratio_and_distance(noun.text, name, ratio_calc=True))
            object += noun.text.lower()
    """
    return object


def version_resolution(mes, mes_speech_parts):
    mes_pos = 0
    mes_obj_count = 0
    mes_id_objects = Objects()
    for noun in mes_speech_parts.nouns:
        mes_id_objects[noun.text + '_' + str(mes_obj_count)] = Object(noun.text.lower())
        mes_id_name = noun.text.lower() + ' '
        for i in range(mes_pos, len(mes) - 1):
            if mes[i] == noun and (len(mes[i].text) > 2) and (
                    mes[i + 1].pos_ == "NUM" or (mes[i + 1].pos_ == "PROPN" and len(mes[i + 1].text) <= 2)):  # No dependencies between words
                mes_id_name += mes[i + 1].text + ' '  # Work Vista lol
                mes_id_objects[noun.text + '_' + str(mes_obj_count)]["version"] = mes[
                    i + 1].text.lower()
                mes_pos = i + 1
        mes_id_name = mes_id_name[:-1]
        mes_obj_count += 1

    for m_obj in vars(mes_id_objects):
        if not hasattr(m_obj, "version"):
            if len(mes_id_objects[m_obj]["name"]) > 2:
                numbers = re.findall(r'\d+', mes_id_objects[m_obj]["name"])
                if numbers:
                    mes_id_objects[m_obj]["name"] = mes_id_objects[m_obj]["name"].replace(numbers[0], '')
                    mes_id_objects[m_obj]["version"] = numbers[0]

    return mes_id_objects


def concurrences_count(mes_id_objects, id_objects, mes_language, doc_language, bool):
    conc_count = 0

    for m_obj in vars(mes_id_objects):
        for obj in vars(id_objects):
            if levenshtein_ratio_and_distance(mes_id_objects[m_obj]["name"].lower(),
                                              id_objects[obj]["name"].lower(),
                                              ratio_calc=True) >= 0.8:
                if hasattr(id_objects[obj], "version"):
                    if hasattr(mes_id_objects[m_obj], "version"):
                        if id_objects[obj]["version"] == mes_id_objects[m_obj]["version"]:
                            del id_objects[obj]
                            conc_count += 1
                            break
                    else:
                        if bool:
                            conc_count += 1
                            break
                        else:
                            break
                else:
                    del id_objects[obj]
                    conc_count += 1
                    break

    if not vars(id_objects):
        return conc_count

    if mes_language == doc_language:

        if mes_language == 'en':
            nlp_m = nlp_en
            nlp_d = nlp_en
        else:
            nlp_m = nlp_de
            nlp_d = nlp_de

        for m_obj in vars(mes_id_objects):
            for obj in vars(id_objects):
                if nlp_m.vocab.get_vector(mes_id_objects[m_obj]["name"])[0] != 0 \
                        and nlp_m.vocab.get_vector(id_objects[obj]["name"])[0] != 0:
                    if cos_sim(nlp_m.vocab.get_vector(mes_id_objects[m_obj]["name"]),
                               nlp_d.vocab.get_vector(id_objects[obj]["name"])) > 0.85:
                        if hasattr(id_objects[obj], "version"):
                            if hasattr(mes_id_objects[m_obj], "version"):
                                if id_objects[obj]["version"] == mes_id_objects[m_obj]["version"]:
                                    del id_objects[obj]
                                    conc_count += 1
                                    break
                            else:
                                if bool:
                                    conc_count += 1
                                    break
                                else:
                                    break
                        else:
                            del id_objects[obj]
                            conc_count += 1
                            break
    else:
        if mes_language == 'en':
            nlp_m = nlp_en
            nlp_d = nlp_de
            translation_matrix = en_de_trans_model.translation_matrix

            space = de_space
        else:
            nlp_m = nlp_de
            nlp_d = nlp_en
            translation_matrix = de_en_trans_model.translation_matrix

            space = en_space

        for m_obj in vars(mes_id_objects):
            for obj in vars(id_objects):
                if nlp_m.vocab.get_vector(mes_id_objects[m_obj]["name"])[0] != 0 \
                        and nlp_m.vocab.get_vector(id_objects[obj]["name"])[0] != 0:
                    print(cos_sim(
                        np.dot(nlp_m.vocab.get_vector(mes_id_objects[m_obj]["name"]), translation_matrix),
                        nlp_d.vocab.get_vector(id_objects[obj]["name"])))
                    if cos_sim(
                            np.dot(nlp_m.vocab.get_vector(mes_id_objects[m_obj]["name"]), translation_matrix),
                            nlp_d.vocab.get_vector(id_objects[obj]["name"])) > 0.5:
                        # TODO Use faster operation with .csv
                        candidates_list = space.most_similar(
                            positive=[np.dot(nlp_m.vocab.get_vector(mes_id_objects[m_obj]["name"]),
                                             translation_matrix)])
                        print(candidates_list)
                        if levenshtein_ratio_and_distance(nlp_d(candidates_list[0][0])[0].lemma_,
                                                          id_objects[obj]["name"].lower(),
                                                          ratio_calc=True) >= 0.8:
                            if hasattr(id_objects[obj], "version"):
                                if hasattr(mes_id_objects[m_obj], "version"):
                                    if id_objects[obj]["version"] == mes_id_objects[m_obj][
                                        "version"]:
                                        del id_objects[obj]
                                        conc_count += 1
                                        break
                                else:
                                    if bool:
                                        conc_count += 1
                                        break
                                    else:
                                        break
                            else:
                                del id_objects[obj]
                                conc_count += 1
                                break

    return conc_count


def df_processing(df, mes, mes_language, doc_language, q_entity, index):
    if mes_language == 'en':
        nlp_m = nlp_en
        #space_m = en_space
    else:
        nlp_m = nlp_de
        #space_m = de_space

    if doc_language == 'en':
        nlp_d = nlp_en
        space_d = en_space
    else:
        nlp_d = nlp_de
        space_d = de_space

    objects = Objects()
    mes_sentence_parts = parts_of_the_sentence()
    mes_speech_parts = parts_of_the_speech()
    mes_sentence_parts, mes_speech_parts, mes_root = sentence_processing(mes, mes_sentence_parts,
                                                                               mes_speech_parts)
    # print("Mes:", mes_sentence_parts.dobj, mes_sentence_parts.pobj, mes_sentence_parts.subj,
    #       mes_sentence_parts.amod)
    # print(mes_speech_parts.nouns, mes_speech_parts.verbs, mes_root, mes_speech_parts.wh)
    # print("Entities:", [(X.text, X.label_) for X in mes.ents])

    mes_no_stop_words = nlp_m(' '.join([str(t) for t in mes if not t.is_stop]))
    #print(mes_no_stop_words)

    name = name_resolution(mes, mes_language, q_entity,  mes_sentence_parts, mes_speech_parts, mes_root)
    objects[name] = Object(name)

    sim_coef = 0.70 #45°
    en_de_sim_coef = 0.50 #60°

    for j_o in df:
        if hasattr(objects, name):
            sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', j_o['question'])

            for sentence in sents: #doc.sents
                sent = nlp_d(sentence)
                df_q_entity = question_resolution(sent, doc_language, True)

                sent_no_stop_words = nlp_d(' '.join([str(t) for t in sent if not t.is_stop]))
                # print(sent_no_stop_words)

                if mes_language == doc_language:
                    if q_entity == df_q_entity:
                        doc_similarity = mes_no_stop_words.similarity(sent_no_stop_words)
                        if doc_similarity >= sim_coef:
                            index.append([doc_similarity, sentence, j_o['question'], j_o['answer']])
                else:
                    word_pairs = pd.read_csv("word_pairs.csv")
                    if q_entity == df_q_entity:
                        mv = sentence_vektor(mes_no_stop_words)
                        sv = sentence_vektor(sent_no_stop_words)
                        if mes_language == 'en':
                            word_list = word_pairs.values.tolist()
                            for word_pair in word_list:
                                word_pair[0], word_pair[1] = word_pair[1], word_pair[0]

                            de_list = []
                            for word_pair in word_list:
                                de_list.append(word_pair[0])

                            for word in mes_no_stop_words:
                                if nlp_de.vocab.get_vector(word.text)[
                                    0] != 0:  # TODO Learn new word vectors from context
                                    if word.text not in de_list:
                                        temp = translator.translate(word.text, src='de', dest='en').text
                                        # result = translate.translate_text(Text=word.text, SourceLanguageCode="de", TargetLanguageCode="en")
                                        # translation = result.get('TranslatedText') #amazon api
                                        if nlp_en.vocab.get_vector(temp)[0] != 0:
                                            temp_pair = [word.text, temp]
                                            df = pd.DataFrame([temp_pair], columns=['de', 'en'])
                                            word_pairs = word_pairs.append(df, ignore_index=True)

                            word_pairs.to_csv("word_pairs.csv", index=False, header=True)
                            de_en_trans_model.train(word_list)
                            mv_t = np.dot(mv, de_en_trans_model.translation_matrix)
                            doc_similarity = cos_sim(mv_t, sv)
                            if doc_similarity >= en_de_sim_coef:
                                index.append([doc_similarity, sentence, j_o['question'], j_o['answer']])
                        else:
                            word_list = word_pairs.values.tolist()
                            for word_pair in word_list:
                                word_pair[0], word_pair[1] = word_pair[1], word_pair[0]

                            de_list = []
                            for word_pair in word_list:
                                de_list.append(word_pair[0])

                            for word in mes_no_stop_words:
                                if nlp_de.vocab.get_vector(word.text)[
                                    0] != 0:  # TODO Learn new word vectors from context
                                    if word.text not in de_list:
                                        temp = translator.translate(word.text, src='de', dest='en').text
                                        # result = translate.translate_text(Text=word.text, SourceLanguageCode="de", TargetLanguageCode="en")
                                        # translation = result.get('TranslatedText') #amazon api
                                        if nlp_en.vocab.get_vector(temp)[0] != 0:
                                            temp_pair = [word.text, temp]
                                            df = pd.DataFrame([temp_pair], columns=['de', 'en'])
                                            word_pairs = word_pairs.append(df, ignore_index=True)

                            word_pairs.to_csv("word_pairs.csv", index=False, header=True)
                            de_en_trans_model.train(word_list)
                            mv_t = np.dot(mv, de_en_trans_model.translation_matrix)
                            doc_similarity = cos_sim(mv_t, sv)
                            if doc_similarity >= en_de_sim_coef:
                                index.append([doc_similarity, sentence, j_o['question'], j_o['answer']])
    index = sorted(index, reverse=True)
    mes_id_objects = version_resolution(mes, mes_speech_parts)

    for candidate in index:
        doc = nlp_d(candidate[2])

        doc_sentence_parts = parts_of_the_sentence()
        doc_speech_parts = parts_of_the_speech()
        doc_sentence_parts, doc_speech_parts, root = sentence_processing(doc, doc_sentence_parts,
                                                                         doc_speech_parts)
        doc_id_objects = version_resolution(doc, doc_speech_parts)

        sent = nlp_d(candidate[1])
        sent_no_stop_words = nlp_d(' '.join([str(t) for t in sent if not t.is_stop]))

        sent_sentence_parts = parts_of_the_sentence()
        sent_speech_parts = parts_of_the_speech()
        sent_sentence_parts, sent_speech_parts, sent_root = sentence_processing(sent, sent_sentence_parts,
                                                                                sent_speech_parts)
        id_objects = version_resolution(sent, sent_speech_parts)
        conc_count = concurrences_count(mes_id_objects, id_objects, mes_language, doc_language, True)

        if conc_count == len(mes_speech_parts.nouns):
            doc_conc_count = concurrences_count(mes_id_objects, doc_id_objects, mes_language, doc_language, False)
            if doc_conc_count == len(doc_speech_parts.nouns):
                # doc_a = nlp(j_o['answer'])
                if q_entity == "yes/no":
                    answer_byte = candidate[2].encode('utf-8')
                    an = np.array([answer_byte])
                    predict = yes_no_model_en.predict(an)
                    if predict[0][0] >= 0:
                        answer = "Yeees. "
                    else:
                        answer = "Nooo. "

                    answer += "According to: " + "question: " + candidate[2] + "; answer: " + candidate[
                        3] + '\n' + str(mes_no_stop_words.similarity(sent_no_stop_words))
                    return answer, True
                else:
                    #answer = 'some_subj' + '\n' #TODO Language spaces need a lot of virtual memory (30 Gb)
                    answer = ''
                    answer_faq = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', j_o['answer'])
                    for ans_sentence in answer_faq:
                        ans = nlp_d(ans_sentence.strip())
                        ans_sentence_parts = parts_of_the_sentence()
                        ans_speech_parts = parts_of_the_speech()
                        ans_sentence_parts, ans_speech_parts, ans_root = sentence_processing(ans,
                                                                                         ans_sentence_parts,
                                                                                         ans_speech_parts)
                        for noun in ans_speech_parts.nouns:
                            for place_word in entity_map[q_entity]:
                                if answer == '':
                                    candidates_list = space_d.most_similar(
                                        positive=[q_entity, noun.text],
                                        negative=[place_word])
                                    print(candidates_list)
                                    for list in candidates_list:
                                        if answer == '':
                                            for place_word in entity_map[q_entity]:
                                                if nlp_d(list[0])[0].lemma_ == place_word:
                                                    answer = noun.text + ' ' + str(list[1]) + '\n'
                                                    break

                    answer += "According to: " + "question: " + candidate[2] + "; answer: " + candidate[
                        3] + '\n' + str(candidate[0])
                    return answer, True
            else:
                par = 'If you have '
                for obj in vars(doc_id_objects):
                    if hasattr(doc_id_objects[obj], "version"):
                        par += doc_id_objects[obj]["name"] + ' ' + doc_id_objects[obj][
                            "version"] + ' '

                if par == 'If you have ':
                    par = ''

                if q_entity == "yes/no":
                    answer_byte = candidate[2].encode('utf-8')
                    an = np.array([answer_byte])
                    predict = yes_no_model_en.predict(an)
                    if predict[0][0] >= 0:
                        answer = par + "it's Yeees. "
                    else:
                        answer = par + "it's Nooo. "

                    answer += "According to: " + "question: " + candidate[2] + "; answer: " + candidate[
                        3] + '\n' + str(candidate[0])
                    return answer, True
                else:
                    return par + "According to: " + "question: " + candidate[2] + "; answer: " + candidate[
                        3] + '\n' + str(candidate[0]), True
        else:
            doc_conc_count = concurrences_count(mes_id_objects, doc_id_objects, mes_language,
                                                doc_language, True)
            if doc_conc_count == len(mes_speech_parts.nouns):
                par = 'If you have '
                for obj in vars(doc_id_objects):
                    if hasattr(doc_id_objects[obj], "version"):
                        par += doc_id_objects[obj]["name"] + ' ' + doc_id_objects[obj][
                            "version"] + ' '

                if par == 'If you have ':
                    par = ''

                if q_entity == "yes/no":
                    answer_byte = candidate[2].encode('utf-8')
                    an = np.array([answer_byte])
                    predict = yes_no_model_en.predict(an)
                    if predict[0][0] >= 0:
                        answer = par + "it's Yeees. "
                    else:
                        answer = par + "it's Nooo. "

                    answer += "According to: " + "question: " + candidate[2] + "; answer: " + candidate[
                        3] + '\n' + str(candidate[0])
                    return answer, True
                else:
                    return par + "According to: " + "question: " + candidate[2] + "; answer: " + candidate[
                        3] + '\n' + str(candidate[0]), True

    return objects[name], False


def main_handler(df, message, mes_language, doc_language):
    mes = language_resolution(message, mes_language)
    # for token in mes:
    #     print(token.text, token.tag_, token.head.text, token.dep_, token.lemma_, token.pos_)

    q_entity = question_resolution(mes, mes_language, True)  # TODO Is a question
    #print("Question_type:", q_entity)
    received = False

    if q_entity:
        most_possible = []
        answer, received = df_processing(df, mes, mes_language, doc_language, q_entity, most_possible)

        if not isinstance(answer, str):
            #pprint(vars(answer))
            if not hasattr(answer, q_entity):
                mes_sentence_parts = parts_of_the_sentence()
                mes_speech_parts = parts_of_the_speech()
                mes_sentence_parts, mes_speech_parts, mes_root = sentence_processing(mes, mes_sentence_parts, mes_speech_parts)

                subj = ''
                name = name_resolution(mes, mes_language, q_entity, mes_sentence_parts, mes_speech_parts, mes_root)
                for token in mes:
                    if token.head.text == name and token.text != name and (
                            token.pos_ == "ADJ" or token.tag_.find("NN") != -1):
                        subj += token.text + ' '
                subj += name

                if most_possible:
                    most_possible = sorted(most_possible, reverse=True)
                    if q_entity == question_entity_map[mes_language][0][1]:
                        answer = "\nThe most similar question is: " + \
                                 most_possible[0][1] + "\nThe answer is: " + most_possible[0][3] + '\n' + str(most_possible[0][0])

                    elif q_entity == question_entity_map[mes_language][3][1]:
                        answer = "\nThe most similar question is: " + \
                                 most_possible[0][1] + "\nThe answer is: " + most_possible[0][3] + '\n' + str(most_possible[0][0])

                    elif q_entity == 'yes/no':
                        answer_byte = str(most_possible[0][0]).encode('utf-8')
                        an = np.array([answer_byte])
                        predict = yes_no_model_en.predict(an)
                        if predict[0][0] >= 0:
                            answer = "Yeees."
                        else:
                            answer = "Nooo."

                        answer += "\nThe most similar question is: " + \
                                 most_possible[0][1] + "\nThe answer is: " + most_possible[0][3] + '\n' + str(most_possible[0][0])
    else:
        answer = "Is this a question?"

    if isinstance(answer, Object):
        answer = "Nothing."

    return answer, received


PORT = 4242
Handler = http.server.SimpleHTTPRequestHandler

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        start = time.time()
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        self.send_response(200)
        # Some predefined functions that show content related information for given words
        message = body.decode("utf-8")
        url = message.split('------WebKitFormBoundary')[2].split('name="page_url"')[1].replace('\n', '').replace('\r', '')
        asin = re.findall(r'[0-9A-Z]{10}', url)[0]
        print(asin)
        message = message.split('------WebKitFormBoundary')[1].split('name="message"')[1].replace('\n', '').replace('\r', '')
        print(message)

        mes = nlp_en(message)
        mes_language = mes._.language["language"]
        doc_language = mes_language
        print(mes_language)

        if mes_language not in language_map:
            mes_language = 'en'
            doc_language = 'en'

        req_start = time.time()
        r = requests.get("https://analytics.liebl.de/aqc?asin=" + asin + "&language=" + mes_language)
        content = r.content.decode("utf-8")
        df = json.loads(content)
        req_end = time.time()

        req = False
        received = False
        answer = "First request time: " + str(req_end - req_start) + " seconds" + '\n'

        if df:
            temp, received = main_handler(df, message, mes_language, doc_language)
            answer += temp
        else:
            req = True
            answer += "FAQ document for " + doc_language + " had not been received" + '\n'
            doc_language = switch_language(doc_language)

            req_start = time.time()
            r = requests.get("https://analytics.liebl.de/aqc?asin=" + asin + "&language=" + doc_language)
            content = r.content.decode("utf-8")
            df = json.loads(content)
            req_end = time.time()

            answer += "Second request time: " + str(req_end - req_start) + " seconds" + '\n'

            if df:
                temp, received = main_handler(df, message, mes_language, doc_language)
                answer += temp
            else:
                answer += "FAQ document for " + doc_language + " had not been received" + '\n'

        if not received and not req:
            doc_language = switch_language(doc_language)
            answer += "\nTrying to receive answer from " + doc_language + '\n'

            req_start = time.time()
            r = requests.get("https://analytics.liebl.de/aqc?asin=" + asin + "&language=" + doc_language)
            content = r.content.decode("utf-8")
            df = json.loads(content)
            req_end = time.time()

            answer += "Second request time: " + str(req_end - req_start) + " seconds" + '\n'

            if df:
                temp, received = main_handler(df, message, mes_language, doc_language)
                answer += temp
            else:
                answer += "FAQ document for " + doc_language + " had not been received" + '\n'

        end = time.time()
        answer += '\n' + "Time taken:" + str(end - start) + " seconds"

        print(answer)

        response = BytesIO()
        response.write(answer.encode('utf-8'))
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(response.getvalue())))
        self.send_header("Access-Control-Allow-Origin", '*')
        self.send_header('Access-Control-Allow-Methods', "POST")
        self.end_headers()
        self.wfile.write(response.getvalue())


# from gensim.models import KeyedVectors

with socketserver.TCPServer(("", PORT), SimpleHTTPRequestHandler) as httpd:
    print("serving at port", PORT)

    # TODO Training works faster than loading i dunno
    # print("Loading translation matrices...")
    # start = time.time()
    # en_de_trans_model = TranslationMatrix.load("en_de_translation_matrix_spacy")
    # de_en_trans_model = TranslationMatrix.load("de_en_translation_matrix_spacy")
    # end = time.time()
    # print(end - start, " seconds")

    print("Loading spacy models...")
    start = time.time()
    nlp_en = spacy.load('en_core_web_md') #sm #md #lg
    nlp_de = spacy.load('de_core_news_md')
    end = time.time()
    print(end - start, " seconds")
    nlp_en.add_pipe(LanguageDetector(), name='language_detector', last=True)

    print("Loading language spaces...")
    start = time.time()
    en_space = Space.build(nlp_en)
    de_space = Space.build(nlp_de)
    end = time.time()
    print(end - start, " seconds")

    word_pairs = pd.read_csv("word_pairs.csv")

    en_de_word_pairs = word_pairs.values.tolist()
    de_en_word_pairs = word_pairs.values.tolist()

    for word_pair in de_en_word_pairs:
        word_pair[0], word_pair[1] = word_pair[1], word_pair[0]

    print("Loading translation matrices...")
    start = time.time()
    en_de_trans_model = TranslationMatrix(nlp_en, nlp_de, word_pairs=en_de_word_pairs)
    de_en_trans_model = TranslationMatrix(nlp_de, nlp_en, word_pairs=de_en_word_pairs)
    end = time.time()
    print(end - start, " seconds")

    yes_no_model_en = tf.keras.models.load_model('yes_no_model')
    #yes_no_model = tf.keras.models.load_model('ja_nein_model')

    translator = Translator()

    httpd.serve_forever()
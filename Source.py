from nltk.tokenize import sent_tokenize
from gensim.models import KeyedVectors
from pprint import pprint
message = "at which floor will Mark and Kevin work today?"

#01.17.2020 Ok.
file = "So we got a lot things to do today. Then i want Julia to organize all the products on the first floor.\
        Kevin and Mark should go to the second floor and take those boxes, which came yesterday.\
        The bolts and screws should lie together. Our client service will be sitting in the basement.\
        We are working as always from 9 a.m. till evening."

damn = ""


import spacy
nlp = spacy.load('en_core_web_sm')
mes = nlp(message)

class Object:
    def __init__(self, name):
        self['name'] = name

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

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

def sentence_processing(doc, sentence_parts, speech_parts):
    root = ''
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)

    for token in doc:
        #print(token.text, token.tag_, token.head.text, token.dep_, token.lemma_)
        if token.dep_ == "dobj":
            sentence_parts.dobj.append(token)

        elif token.dep_ == "pobj":
            sentence_parts.pobj.append(token)

        elif token.dep_ == "nsubj":
            sentence_parts.subj.append(token)

        elif token.dep_ == "amod":
            sentence_parts.amod.append(token)

        elif token.dep_ == "npadvmod" or token.dep_ == "nummod":
            sentence_parts.num.append(token)

        elif token.dep_ == "ROOT" and "VB" in token.tag_:
            root = token.text

        if "NN" in token.tag_:
            if token.text not in entities:
                speech_parts.nouns.append(token)

        elif "VB" in token.tag_:
            speech_parts.verbs.append(token)

    print(sentence_parts.dobj, sentence_parts.pobj, sentence_parts.subj, sentence_parts.amod)
    print(speech_parts.nouns, speech_parts.verbs, root)
    return(sentence_parts, speech_parts, root)


def time_resolution(doc, time_entities, name, speech_parts):
    time_stamp = None
    if len(time_entities) == 1:
        time_stamp = time_entities[0]
        return time_stamp

    elif len(time_entities) > 1:
        for entity in time_entities:
            for ent in doc.ents:
                if ent.label_ == "DATE" and entity == ent.text:
                    for word in doc:
                        if word.text == ent.text:
                            core = word
                            word = word.head
                            for noun in speech_parts.nouns:
                                if word.text == noun.text and word.text == name:
                                    time_stamp = core.lemma_
                                    return time_stamp
    if len(time_entities) > 1:
        for entity in time_entities:
            for ent in doc.ents:
                if ent.label_ == "DATE" and entity != ent.text:
                    time_stamp = entity
                    return time_stamp

    return time_stamp


def answer_generation(doc, place):
    adj = ''
    for word in doc:
        if word.head.text == place:
            adj += word.text + ' '
    adj += place
    return adj


def place_resolution(doc, speech_parts, root):
    place = None
    lemma_nouns = []
    for noun in speech_parts.nouns:
        lemma_nouns.append(noun.lemma_)

    print(1, word_model.most_similar(positive=["movement", root], negative=["go"]))
    word_set = word_model.most_similar(positive=["movement", root], negative=["go"])
    for wd in word_set:
        if nlp(wd[0])[0].lemma_ == "movement":
            for word in speech_parts.nouns:
                core = word
                word = word.head
                while word.text != root:
                    word = word.head
                else:
                    place = core.lemma_
                    place = answer_generation(doc, place)
                    return place

    if place is None and root:
        for word in speech_parts.nouns:
            print(2, word_model.most_similar(positive=["place", word.text], negative=["room"]))
            word_set = word_model.most_similar(positive=["place", word.text], negative=["room"])
            for wd in word_set:
                if nlp(wd[0])[0].lemma_ == "place":
                    place = word.lemma_
                    place = answer_generation(doc, place)
                    return place

    return place

def file_processing(file, ents, mes_sentence_parts):
    time_stamp = None
    objects = []
    time_entities = []
    sentences = sent_tokenize(file)
    for sentence in sentences:
        doc = nlp(sentence)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                time_entities.append(ent.text)
        for word in doc:
            if word.text == mes_sentence_parts.subj[0].text: #synonym
                print([(X.text, X.label_) for X in doc.ents])
                sentence_parts = parts_of_the_sentence()
                speech_parts = parts_of_the_speech()
                sentence_parts, speech_parts, root = sentence_processing(doc, sentence_parts, speech_parts)
                c_count = 0
                for ent_mes in ents:
                    for ent_doc in doc.ents:
                        if ent_doc.text == ent_mes.text:
                            c_count += 1
                            print("Found a concurrence", ent_doc.text, ent_doc.label_)
                            if ent_doc.label_ == "PERSON":
                                object = Object(ent_doc.text)
                                object["date"] = time_resolution(doc, time_entities, object["name"], speech_parts)
                                object["place"] = place_resolution(doc, speech_parts, root)
                                objects.append(object)
    return objects

def is_question():
    # dict_model = KeyedVectors.load_word2vec_format("C:\\Users\\Konstantin\\PycharmProjects\\without_server\\dict2vec-vectors-dim100.vec", binary=False)
    print(10, word_model.most_similar(positive=["question", "what"], negative=["which"]))  # if its question+
    print(10, word_model.most_similar(positive=["question", "what"], negative=["where"]))



print([(X.text, X.label_) for X in mes.ents])

sentence_parts = parts_of_the_sentence()
speech_parts = parts_of_the_speech()
sentence_parts, speech_parts, root = sentence_processing(mes, sentence_parts, speech_parts)

word_model = KeyedVectors.load_word2vec_format(
        "C:\\Users\\Konstantin\\PycharmProjects\\without_server\\glove.6B.50d.txt",
        binary=False)

persons = file_processing(file, mes.ents, sentence_parts)
for dead in persons:
    pprint(vars(dead))

#from spacy import displacy
#mes = nlp("Kevin and Mark should go to the second floor and take those boxes, which came yesterday.")
#sentence_spans = list(mes.sents)
#displacy.serve(sentence_spans, style='dep')
#displacy.serve(doc, style="ent")
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

filename = "text_corpus/input.txt"
doc = load_doc(filename)
doc = doc[:len(doc) // 32]

import string
# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens

# clean document
tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

# organize into sequences of tokens
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)
print('Total Sequences: %d' % len(sequences))

# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# save sequences to file
out_filename = 'republic_sequences.txt'
save_doc(sequences, out_filename)

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# load
in_filename = 'republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

from keras.preprocessing.text import Tokenizer
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy
import os
import tensorflow as tf

class_names = ['$', '-LRB-', '-RRB-', '.', 'ADD', 'AFX', 'CC', 'CD', 'DT', 'EX', 'FW', 'HYPH', 'IN', 'JJ', 'JJR',
                   'JJS', 'LS', 'MD', 'NFP', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS',
                   'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX', '_SP']

def generate_answer(persons):
    answer = ''
    i = 1
    if len(persons) > 1:
        for dead in persons:
            answer += dead["name"] + ' '
            if i == 1:
                answer += 'and' + ' '
                i += 1
    else:
        pass
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    slots_pos = []
    for dead in persons:
        for v in vars(dead):
            if v != 'name':
                temp = nlp(dead[v])
                for token in temp:
                    if token.dep_ == 'ROOT':
                        slot_map = []
                        slot_map.append(dead[v])
                        slot_map.append(token.tag_)
                        slots_pos.append(slot_map)
                        break
        break#person
    print(slots_pos)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # load cleaned text sequences
    in_filename = 'republic_sequences.txt'
    doc = load_doc(in_filename)
    lines = doc.split('\n')

    import random
    from random import randint
    # select a seed text
    random.seed(30)
    seed_text = lines[randint(0, len(lines))]
    seed_text = seed_text.rsplit(' ', 1)[0]
    print(seed_text + '\n')
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    tokenizer.fit_on_texts(file)
    seed_word = encoded[:length][0]
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    encoded_pos = [[class_names.index('NN')]]
    while slots_pos != []:
        for slot in slots_pos:
            encoded_pos = numpy.array(encoded_pos)
            encoded_pos = numpy.reshape(encoded_pos, (encoded_pos.shape[0], 1, encoded_pos.shape[1]))

            export_dir = os.path.join(os.getcwd(), 'model_pos')
            # predict probabilities for each word
            with tf.Session(graph=tf.Graph()) as sess:
                tf.saved_model.loader.load(sess, ['serve'], export_dir)
                graph = tf.get_default_graph()
                tensor_input = sess.graph.get_tensor_by_name('lstm_input:0')
                tensor_output = sess.graph.get_tensor_by_name('dense_1/Softmax:0')
                y_pred = sess.run(tensor_output, feed_dict={tensor_input: encoded_pos})
                # print(y_pred)

            maxElement = numpy.amax(y_pred[0])
            print('Max element from Numpy Array : ', maxElement)

            # Get the index of maximum element in numpy array
            print(sorted([(x, i) for (i, x) in enumerate(y_pred[0])], reverse=True)[:3])
            result_pos = numpy.where(y_pred[0] == numpy.amax(y_pred[0]))
            print('List of Indices of maximum pos_element :', result_pos[0])
            if slot[1] == class_names[result_pos[0][0]]:
                answer += slot[0] + ' '
                seed_word = slot[0]
                slots_pos.pop(0)
            else:
                # integer encode sequences of words
                seed_text += ' ' + tokens[seed_word]
                encoded = tokenizer.texts_to_sequences([seed_text])[0]
                encoded.pop()
                encoded_arr = []
                encoded_arr.append(encoded)
                encoded = numpy.array(encoded_arr)
                export_dir = os.path.join(os.getcwd(), 'model_gen')
                # predict probabilities for each word
                with tf.Session(graph=tf.Graph()) as sess:
                    tf.saved_model.loader.load(sess, ['serve'], export_dir)
                    graph = tf.get_default_graph()
                    tensor_input = sess.graph.get_tensor_by_name('embedding_input:0')
                    tensor_output = sess.graph.get_tensor_by_name('dense_1/Softmax:0')
                    y_pred = sess.run(tensor_output, feed_dict={tensor_input: encoded})
                    # print(y_pred)
                    result = numpy.where(y_pred[0] == numpy.amax(y_pred[0]))
                    print('List of Indices of maximum element :', result[0])
                    print(tokens[result[0][0]])
                answer += tokens[result[0][0]] + ' '
                seed_word = result[0][0]

                encoded_pos = [[class_names.index(nlp(tokens[result[0][0]])[0].tag_)]]
                seed_text = seed_text.split(' ', 1)[1]
                print(seed_text)

    return answer

answer = generate_answer(persons)
print(answer)
message = " at which floor will Mark and Kevin work today?"

file = "01.17.2020 Ok. So we got a lot things to do today. Kevin and Mark should go to the second floor and take those boxes, which came yesterday., \
Then i want Julia to organize all the products on the first floor. The bolts and screws should lie together. Our client service will be sitting in the basement., \
We are working as always from 9 a.m. till evening."

import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(message)

#print([token.text for token in doc])
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

sentence_parts = parts_of_the_sentence()
speech_parts = parts_of_the_speech()
root = ''

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
        speech_parts.nouns.append(token.lemma_)

    elif "VB" in token.tag_:
        speech_parts.verbs.append(token.lemma_)

print(sentence_parts.dobj, sentence_parts.pobj, sentence_parts.subj, sentence_parts.amod, sentence_parts.num)
print(speech_parts.nouns, speech_parts.verbs, root)

""""""
def place_resolution(ent, subj):
    pass

from gensim.models import KeyedVectors
word_model = KeyedVectors.load_word2vec_format("C:\\Users\\Konstantin\\PycharmProjects\\without_server\\glove.6B.100d.txt", binary=False)#"dict2vec-vectors-dim100.vec"#glove.6B.50d.txt
print(2, word_model.most_similar(positive=["5"]))
print(2, word_model.most_similar(positive=["2"]))
print(word_model.similarity("5", "2"))

print(111, word_model.most_similar(positive=["man", "royalty"]))
print(111, word_model.most_similar(positive=["king", "woman"], negative=["queen"]))

print(2, word_model.most_similar(positive=["mac"]))
print(2, word_model.most_similar(positive=["computer"]))
print(9, word_model.most_similar(positive=["animal", "mac"], negative=["dog"]))  # hypernum

print(0, word_model.most_similar(positive=["place", "floor"], negative=["room"]))
print(0, word_model.most_similar(positive=["place", "floor"], negative=["place"]))

print(1,word_model.most_similar(positive=["group", "of"], negative=["relatives"]))
print(2,word_model.most_similar(positive=["relatives"]))
print(3,word_model.most_similar(positive=["relatives", "of"], negative=["group"]))

print(4.1,word_model.most_similar(positive=["good", "female"], negative=["bad"]))   #antonim
print(4.1, word_model.most_similar(positive=["bad", "female"], negative=["good"]))  #antonim

print(4.2, word_model.most_similar(positive=["good", "male"], negative=["bad"]))  # antonim
print(4.2, word_model.most_similar(positive=["bad", "male"], negative=["good"]))  # antonim

print(4.2, word_model.most_similar(positive=["male", "female"], negative=["good"]))  # antonim

print(4.3, word_model.most_similar(positive=["good", "white"], negative=["bad"]))  # antonim
print(4.3, word_model.most_similar(positive=["bad", "white"], negative=["good"]))  # antonim

print(4.4, word_model.most_similar(positive=["good", "black"], negative=["bad"]))  # antonim
print(4.4, word_model.most_similar(positive=["bad", "black"], negative=["good"]))  # antonim

print(4.4, word_model.most_similar(positive=["go"]))  #verb

print(5,word_model.most_similar(positive=["spouse", "female"], negative=["male"]))
print(6,word_model.most_similar(positive=["spouse", "female"]))
print(7,word_model.most_similar(positive=["spouse", "male"]))
print(8,word_model['female'])
print(9,word_model.most_similar(positive=["animal", "car"], negative=["dog"])) #hypernum
print(10, word_model.most_similar(positive=["animal", "cat"], negative=["dog"]))  #if its animal+
####################################################################################################

print(666, word_model.most_similar(positive=["animal", "russia"], negative=["dog"]))
print(666, word_model.most_similar(positive=["animal", "russia"], negative=["pig"]))
print(666, word_model.most_similar(positive=["vehicle", "russia"], negative=["car"]))

print(666, word_model.most_similar(positive=["animal", "germany"], negative=["dog"]))
print(666, word_model.most_similar(positive=["animal", "germany"], negative=["pig"]))
print(666, word_model.most_similar(positive=["vehicle", "germany"], negative=["car"]))
print(777, word_model.most_similar(positive=["color", "animal", "germany"], negative=["red", "dog"]))

print(777, word_model.most_similar(positive=["vehicle", "animal", "germany"], negative=["car", "dog"]))
print(777, word_model.most_similar(positive=["vehicle", "fruit", "animal", "germany"],
                                       negative=["car", "apple", "dog"]))
print(777, word_model.most_similar(positive=["color", "animal", "germany"], negative=["red", "dog"]))
print(777, word_model.most_similar(positive=["color", "germany"], negative=["black"]))
print(777, word_model.most_similar(positive=["color", "germany"], negative=["red"]))

print(666, word_model.most_similar(positive=["animal", "room"], negative=["dog"]))
print(777, word_model.most_similar(positive=["vehicle", "animal", "room"], negative=["car", "dog"]))
print(777, word_model.most_similar(positive=["vehicle", "fruit", "animal", "room"],
                                       negative=["car", "apple", "dog"]))

print(666, word_model.most_similar(positive=["animal", "floor"], negative=["dog"]))
print(777, word_model.most_similar(positive=["vehicle", "animal", "floor"], negative=["car", "dog"]))
print(777, word_model.most_similar(positive=["vehicle", "fruit", "animal", "floor"],
                                       negative=["car", "apple", "dog"]))

print(666, word_model.most_similar(positive=["animal", "bottom"], negative=["dog"]))
print(777, word_model.most_similar(positive=["vehicle", "animal", "bottom"], negative=["car", "dog"]))
print(777, word_model.most_similar(positive=["vehicle", "fruit", "animal", "bottom"],
                                       negative=["car", "apple", "dog"]))

print(666, word_model.most_similar(positive=["animal", "spot"], negative=["dog"]))
print(777, word_model.most_similar(positive=["vehicle", "animal", "spot"], negative=["car", "dog"]))
print(777, word_model.most_similar(positive=["vehicle", "fruit", "animal", "spot"],
                                       negative=["car", "apple", "dog"]))

print(666, word_model.most_similar(positive=["dog", "place"], negative=["animal"]))
print(555, word_model.most_similar(positive=["car", "dog", "place"], negative=["vehicle", "animal"]))
print(555, word_model.most_similar(positive=["car", "apple", "dog", "place"],
                                       negative=["vehicle", "fruit", "animal"]))

print(666, word_model.most_similar(positive=["dog", "spot"], negative=["animal"]))
print(555, word_model.most_similar(positive=["car", "dog", "spot"], negative=["vehicle", "animal"]))
print(555, word_model.most_similar(positive=["car", "apple", "dog", "spot"],
                                       negative=["vehicle", "fruit", "animal"]))

print(666, word_model.most_similar(positive=["dog", "bottom"], negative=["animal"]))
print(555, word_model.most_similar(positive=["car", "dog", "bottom"], negative=["vehicle", "animal"]))
print(555, word_model.most_similar(positive=["car", "apple", "dog", "bottom"],
                                       negative=["vehicle", "fruit", "animal"]))

print(666, word_model.most_similar(positive=["dog", "room"], negative=["animal"]))
print(555, word_model.most_similar(positive=["car", "dog", "room"], negative=["vehicle", "animal"]))
print(555, word_model.most_similar(positive=["car", "apple", "dog", "room"],
                                       negative=["vehicle", "fruit", "animal"]))

print(666, word_model.most_similar(positive=["dog", "floor"], negative=["animal"]))
print(555, word_model.most_similar(positive=["car", "dog", "floor"], negative=["vehicle", "animal"]))
print(555, word_model.most_similar(positive=["car", "apple", "dog", "floor"],
                                       negative=["vehicle", "fruit", "animal"]))

print(666, word_model.most_similar(positive=["dog", "country"], negative=["animal"]))
print(555, word_model.most_similar(positive=["car", "dog", "country"], negative=["vehicle", "animal"]))
print(555, word_model.most_similar(positive=["car", "apple", "dog", "country"],
                                       negative=["vehicle", "fruit", "animal"]))

####################################################################################################
print(10, word_model.most_similar(positive=["animal", "cat"], negative=["animal"]))
print(10, word_model.most_similar(positive=["cat"]))
print(10, word_model.most_similar(positive=["animal", "car"], negative=["animal"]))

print(11, word_model.most_similar(positive=["dog", "vehicle"], negative=["animal"]))  #hypornum
print(12, word_model.most_similar(positive=["animal", "car"], negative=["vehicle"]))  #hypornum

print(13, word_model.most_similar(positive=["air", "water"], negative=["breathe"]))
print(13, word_model.most_similar(positive=["breathe", "water"], negative=["air"]))  #analogi
print(13, word_model.most_similar(positive=["breathe", "drink"], negative=["air"]))

"""
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

    words = []
    classes = []
    documents = []
    ignore_words = ['?']
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern)
            # add to our words list
            words.extend(w)
            # add to documents in our corpus
            documents.append((w, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    # remove duplicates
    classes = sorted(list(set(classes)))

    print(len(documents), "documents")
    print(len(classes), "classes", classes)
    print(len(words), "unique stemmed words", words)

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])

print(train_x)
print(train_y)

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)

from PyDictionary import PyDictionary
dictionary = PyDictionary()
meaning = dictionary.meaning(sentence_parts.pobj[0].text)["Noun"]
print(meaning)

#model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
#model.save('model.tflearn')

# save all of our data structures
#import pickle
#pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )
"""

doc = nlp(file)
for token in doc:
    print(token.text, token.tag_, token.head.text, token.dep_, token.lemma_)

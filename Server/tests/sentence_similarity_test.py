import numpy as np
import spacy
import time

from translation_matrix import TranslationMatrix

print("Loading models...")
start = time.time()
en_model = spacy.load('en_core_web_md')
de_model = spacy.load('de_core_news_md')
end = time.time()
print(end-start, " seconds")

# print("Loading models...")
# start = time.time()
# en_model = KeyedVectors.load_word2vec_format("word2vec.model", binary=True)
# de_model = KeyedVectors.load_word2vec_format("word2vec_german.model", binary=True)
# end = time.time()
# print(end-start, " seconds")

########################################################################################################################

from numpy.linalg import norm
def cos_sim(sv_1, sv_2):
    return np.dot(sv_1, sv_2)/(norm(sv_1)*norm(sv_2))

def sentence_vektor(doc):
    sv = 0
    for token in doc:
        sv += token.vector
    return sv / len(doc)

sent_1 = en_model("Where can i buy a phone?")
sent_no_stop_words_1 = en_model(' '.join([str(t) for t in sent_1 if not t.is_stop]))

sent_2 = en_model("At what place is it possible to get a phone?")
sent_no_stop_words_2 = en_model(' '.join([str(t) for t in sent_2 if not t.is_stop]))

print("Similarity of first english and second english sentence: ", sent_no_stop_words_1.similarity(sent_no_stop_words_2))#0.82710129

vec_1 = sentence_vektor(sent_no_stop_words_1)
vec_2 = sentence_vektor(sent_no_stop_words_2)

print("Custom similarity of first english and second english sentence: ", cos_sim(vec_1, vec_2))

sent_3 = de_model("Wo kann ich ein Handy kaufen?")
sent_no_stop_words_3 = de_model(' '.join([str(t) for t in sent_3 if not t.is_stop]))

sent_4 = de_model("An welchem Platz ist es möglich ein Handy zu bekommen?")
sent_no_stop_words_4 = de_model(' '.join([str(t) for t in sent_4 if not t.is_stop]))

print("Similarity of first german and second german sentence: ", sent_no_stop_words_3.similarity(sent_no_stop_words_4))#0.82186899

vec_3 = sentence_vektor(sent_no_stop_words_3)
vec_4 = sentence_vektor(sent_no_stop_words_4)

print("Custom similarity of first german and second german sentence: ", cos_sim(vec_3, vec_4))

print("Similarity of first english and first german sentence: ", sent_no_stop_words_1.similarity(sent_no_stop_words_3))
print("Similarity of first english and second german sentence: ", sent_no_stop_words_1.similarity(sent_no_stop_words_4))

print("Similarity of second english and first german sentence: ", sent_no_stop_words_2.similarity(sent_no_stop_words_3))
print("Similarity of second english and second german sentence: ", sent_no_stop_words_2.similarity(sent_no_stop_words_4))

en_de_word_pairs = [("area", "Bereich"), ("place", "Ort"), ("spot", "Stelle"), ("room", "Raum"), ("floor", "Boden"), ("side","Seite"),
              ("buy", "kaufen"), ("possible", "möglich"), ("get", "bekommen"), ("work", "Arbeit")]

de_en_word_pairs = [("Bereich", "area"), ("Ort", "place"), ("Stelle", "spot"), ("Raum", "room"), ("Boden", "floor"), ("Seite", "side"),
              ("kaufen", "buy"), ("möglich", "possible"), ("bekommen", "get"), ("Arbeit", "work")]


########################################################################################################################

trans_model = TranslationMatrix(en_model, de_model, word_pairs=en_de_word_pairs)
trans_model.save("en_de_translation_matrix_spacy")  # save model to file

trans_model = TranslationMatrix(de_model, en_model, word_pairs=de_en_word_pairs)
trans_model.save("de_en_translation_matrix_spacy")  # save model to file

#trans_model = TranslationMatrix.load("en_de_translation_matrix_spacy")
#print(trans_model.translation_matrix)

de_vec_1 = np.dot(vec_1, trans_model.translation_matrix)
de_vec_2 = np.dot(vec_2, trans_model.translation_matrix)

print("Similarity of first transformed english and first german sentence: ", cos_sim(de_vec_1, vec_3))
print("Similarity of first transformed english and second german sentence: ", cos_sim(de_vec_1, vec_4))

print("Similarity of second transformed english and first german sentence: ", cos_sim(de_vec_2, vec_3))
print("Similarity of second transformed english and second german sentence: ", cos_sim(de_vec_2, vec_4))

sv_1 = 0
for token in sent_no_stop_words_1:
    sv_1 += np.dot(token.vector, trans_model.translation_matrix)
sv_1 = sv_1 / len(sent_no_stop_words_1)

print("Similarity of first transformed english and first german sentence if we transform word by word: ", cos_sim(sv_1, vec_3))
print("Similarity of first transformed english and second german sentence if we transform word by word: ", cos_sim(sv_1, vec_4))
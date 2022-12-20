import time
import spacy
import pandas as pd
import numpy as np
from numpy.linalg import norm

import boto3
from googletrans import Translator
translator = Translator()

#translate = boto3.client(service_name='translate', region_name='eu-west-1', use_ssl=True)

from translation_matrix import TranslationMatrix
from translation_matrix import Space

print("Loading translation matrices...")
start = time.time()
en_de_trans_model = TranslationMatrix.load("en_de_translation_matrix_spacy")
end = time.time()
print(end - start, " seconds")

word_pairs = pd.read_csv("word_pairs.csv")

def sentence_vektor(doc):
    sv = 0
    for token in doc:
        if token.text in en_model.vocab:#delete this
            sv += token.vector
        else:
            pass
    return sv / len(doc)

print("Loading models...")
start = time.time()
en_model = spacy.load('en_core_web_md')
de_model = spacy.load('de_core_news_md')
end = time.time()
print(end-start, " seconds")

en_sent = "Will there be a rose gold version of this adapter soon?"
de_sent = "Wird es eine pink und goldene Version dieses Adapters sein?"

s_1 = en_model(en_sent)
s_2 = de_model(de_sent)

s_1 = en_model(' '.join([str(t) for t in s_1 if not t.is_stop]))
s_2 = de_model(' '.join([str(t) for t in s_2 if not t.is_stop]))

print(s_1.similarity(s_2))

sv_1 = sentence_vektor(s_1)
sv_2 = sentence_vektor(s_2)

def cos_sim(sv_1, sv_2):
    return np.dot(sv_1, sv_2)/(norm(sv_1)*norm(sv_2))

sv_t_1 = np.dot(sv_1, en_de_trans_model.translation_matrix)

print(cos_sim(sv_t_1, sv_2))
for word in s_1:
    word_list = word_pairs.values.tolist()
    en_list = []
    for word_pair in word_list:
        en_list.append(word_pair[0])

    if word.text not in en_list:
        temp = translator.translate(word.text, dest='de').text
        # result = translate.translate_text(Text=word.text, SourceLanguageCode="en", TargetLanguageCode="de")
        # translation = result.get('TranslatedText')
        if temp in de_model.vocab and word.text in en_model.vocab:
            temp_pair = [word.text, temp]
            print(temp_pair)
            df = pd.DataFrame([temp_pair], columns=['en', 'de'])
            word_pairs = word_pairs.append(df, ignore_index=True)

word_pairs.to_csv("word_pairs.csv", index=False, header=True)
word_list = word_pairs.values.tolist()

print("Train matrix...")
start = time.time()
en_de_trans_model.train(word_list)#8 sec
end = time.time()
print(end - start, " seconds")

sv_t_1 = np.dot(sv_1, en_de_trans_model.translation_matrix)

print(cos_sim(sv_t_1, sv_2))
print("Train matrix...")
start = time.time()
en_de_trans_model.save("translation_matrix")
end = time.time()
print(end - start, " seconds")
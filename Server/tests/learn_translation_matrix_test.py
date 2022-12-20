import numpy as np
import spacy
import time

import pandas as pd

from translation_matrix import TranslationMatrix
from translation_matrix import Space

from numpy import linalg

print("Loading models...")
start = time.time()
en_model = spacy.load('en_core_web_md')
de_model = spacy.load('de_core_news_md')
end = time.time()
print(end-start, " seconds")

word_pairs = pd.read_csv("word_pairs.csv")

en_de_word_pairs = word_pairs.values.tolist()
de_en_word_pairs = word_pairs.values.tolist()

for word_pair in de_en_word_pairs:
    word_pair[0], word_pair[1] = word_pair[1], word_pair[0]

print("Training en->de translation matrix...")
start = time.time()
en_de_trans_model = TranslationMatrix(en_model, de_model, word_pairs=en_de_word_pairs)
en_de_trans_model.save("en_de_translation_matrix_spacy")  # save model to file
end = time.time()
print(end-start, " seconds")

#print(en_de_trans_model.translation_matrix)

print("Training de->en translation matrix...")
start = time.time()
de_en_trans_model = TranslationMatrix(de_model, en_model, word_pairs=de_en_word_pairs)
de_en_trans_model.save("de_en_translation_matrix_spacy")
end = time.time()
print(end-start, " seconds")

#print(linalg.inv(en_de_trans_model.translation_matrix))
#print(de_en_trans_model.translation_matrix)
import pandas as pd
import json
import gzip
from gensim.models import KeyedVectors

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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

df = getDF('qa_Electronics.json.gz')
stop_words = set(stopwords.words('english'))

answers = open("input_vectors_words.txt", "w")
answers_types = open("answers_types.txt", "w")

word_model = KeyedVectors.load_word2vec_format(
       "C:\\Users\\Konstantin\\PycharmProjects\\without_server\\dict2vec-vectors-dim100.vec",
       binary=False)

#if 'word' in word_vectors.vocab
"""
for question in df.values:
    j_o = json.loads(question[0])
    if j_o['questionType'] == 'yes/no':
        if j_o['answerType'] == 'Y':
            answers_types.write('1' + '\n')
            answers.write('\n')
            answers.write(j_o['answer'])
        elif j_o['answerType'] == 'N':
            answers_types.write('0' + '\n')
            answers.write('\n')
            answers.write(j_o['answer'])
"""


answers.write('[')

for question in df.values:
    j_o = json.loads(question[0])
    if j_o['questionType'] == 'yes/no':
        if j_o['answerType'] == 'Y':
            word_tokens = word_tokenize(j_o['answer'])
            if word_tokens:
                answers_types.write('1' + '\n')
                answers.write('[')
                for w in word_tokens:
                    if w not in stop_words and w != '.' and w != ',':
                        if w in word_model.vocab:
                            answers.write('[')
                            for x in word_model[w]:
                                answers.write("%.3f" % x)
                            answers.write(']')
                answers.write(']')
            else:
                print("Ding.")
        elif j_o['answerType'] == 'N':
            word_tokens = word_tokenize(j_o['answer'])
            if word_tokens:
                answers_types.write('0' + '\n')
                answers.write('[')
                for w in word_tokens:
                    if w not in stop_words and w != '.' and w != ',':
                        if w in word_model.vocab:
                            answers.write('[')
                            for x in word_model[w]:
                                answers.write("%.3f" % x)
                            answers.write(']')
                answers.write(']')
            else:
                print("Ding.")


answers.write(']')
answers.close()
answers_types.close()

"""
from ast import literal_eval
print(literal_eval(string))
"""
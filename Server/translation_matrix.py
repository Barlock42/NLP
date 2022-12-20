from collections import OrderedDict
from six import string_types
from gensim import utils
import numpy as np
import warnings

from numbers import Integral
from numpy import dot, float32 as REAL, array, ndarray, sqrt, newaxis
from gensim import matutils

def _l2_norm(m, replace=False):
    dist = sqrt((m ** 2).sum(-1))[..., newaxis]
    if replace:
        m /= dist
        return m
    else:
        return (m / dist).astype(REAL)

class Space(object):

    def __init__(self, matrix, index2word):
        self.mat = matrix
        self.index2word = index2word
        self.norm_vectors = []

        # build a dict to map word to index
        self.word2index = {}
        for idx, word in enumerate(self.index2word):
            self.word2index[word] = idx

        for word in self.index2word:
            self.norm_vectors.append(_l2_norm(self.mat[self.word2index[word]]))

    @classmethod
    def build(cls, lang_vec, lexicon=None):
        # `words` to store all the word that
        # `mat` to store all the word vector for the word in 'words' list
        words = []
        mat = []
        if lexicon is not None:
            # if the lexicon is not provided, using the all the words as default
            for item in lexicon:
                words.append(item)
                mat.append(lang_vec.vocab.get_vector(item))

        else:
            #i = 0
            for item in lang_vec.vocab.strings:
                words.append(item)
                #i += 1
                # token = lang_vec(item)
                # mat.append(token.vector)#works_too_slow
                mat.append(lang_vec.vocab.get_vector(item))
                #print("%0.8f" % (i / len(lang_vec.vocab.strings)))

        return Space(mat, words)

    def normalize(self):
        """Normalize the word vector's matrix."""
        self.mat = self.mat / np.sqrt(np.sum(np.multiply(self.mat, self.mat), axis=1, keepdims=True))

    def most_similar(self, positive=None, negative=None, topn=10, indexer=None):
        if isinstance(topn, Integral) and topn < 1:
            return []

        if positive is None:
            positive = []
        if negative is None:
            negative = []

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [
            (word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in positive
        ]
        negative = [
            (word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in negative
        ]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            else:
                # print(space.mat[space.word2index[word]])
                mean.append(weight * _l2_norm(self.mat[self.word2index[word]]))
                if word in self.index2word:
                    all_words.add(self.word2index[word])
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        if indexer is not None and isinstance(topn, int):
            return indexer.most_similar(mean, topn)

        limited = self.norm_vectors
        dists = dot(limited, mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]


########################################################################################################################

class TranslationMatrix(utils.SaveLoad):

    def __init__(self, source_lang_vec, target_lang_vec, word_pairs=None, random_state=None):
        self.source_word = None
        self.target_word = None
        self.source_lang_vec = source_lang_vec
        self.target_lang_vec = target_lang_vec

        self.random_state = utils.get_random_state(random_state)
        self.translation_matrix = None
        self.source_space = None
        self.target_space = None

        if word_pairs is not None:
            if len(word_pairs[0]) != 2:
                raise ValueError("Each training data item must contain two different language words.")
            self.train(word_pairs)

    def train(self, word_pairs):

        self.source_word, self.target_word = zip(*word_pairs)

        self.source_space = Space.build(self.source_lang_vec, set(self.source_word))
        self.target_space = Space.build(self.target_lang_vec, set(self.target_word))

        self.source_space.normalize()
        self.target_space.normalize()

        m1 = self.source_space.mat[[self.source_space.word2index[item] for item in self.source_word], :]
        m2 = self.target_space.mat[[self.target_space.word2index[item] for item in self.target_word], :]

        self.translation_matrix = np.linalg.lstsq(m1, m2, -1)[0]

    def apply_transmat(self, words_space):
        return Space(np.dot(words_space.mat, self.translation_matrix), words_space.index2word)

    def translate(self, source_words, topn=5, gc=0, sample_num=None, source_lang_vec=None, target_lang_vec=None):

        if isinstance(source_words, string_types):
            # pass only one word to translate
            source_words = [source_words]

        # If the language word vector not provided by user, use the model's
        # language word vector as default
        if source_lang_vec is None:
            warnings.warn(
                "The parameter source_lang_vec isn't specified, "
                "use the model's source language word vector as default."
            )
            source_lang_vec = self.source_lang_vec

        if target_lang_vec is None:
            warnings.warn(
                "The parameter target_lang_vec isn't specified, "
                "use the model's target language word vector as default."
            )
            target_lang_vec = self.target_lang_vec

        # If additional is provided, bootstrapping vocabulary from the source language word vector model.
        if gc:
            if sample_num is None:
                raise RuntimeError(
                    "When using the globally corrected neighbour retrieval method, "
                    "the `sample_num` parameter(i.e. the number of words sampled from source space) must be provided."
                )
            lexicon = set(source_lang_vec.index2word)
            addition = min(sample_num, len(lexicon) - len(source_words))
            lexicon = self.random_state.choice(list(lexicon.difference(source_words)), addition)
            source_space = Space.build(source_lang_vec, set(source_words).union(set(lexicon)))
        else:
            source_space = Space.build(source_lang_vec, source_words)
        target_space = Space.build(target_lang_vec, )

        # Normalize the source vector and target vector
        source_space.normalize()
        target_space.normalize()

        # Map the source language to the target language
        mapped_source_space = self.apply_transmat(source_space)

        # Use the cosine similarity metric
        sim_matrix = -np.dot(target_space.mat, mapped_source_space.mat.T)

        # If `gc=1`, using corrected retrieval method
        if gc:
            srtd_idx = np.argsort(np.argsort(sim_matrix, axis=1), axis=1)
            sim_matrix_idx = np.argsort(srtd_idx + sim_matrix, axis=0)
        else:
            sim_matrix_idx = np.argsort(sim_matrix, axis=0)

        # Translate the words and for each word return the `topn` similar words
        translated_word = OrderedDict()
        for idx, word in enumerate(source_words):
            translated_target_word = []
            # Search the most `topn` similar words
            for j in range(topn):
                map_space_id = sim_matrix_idx[j, source_space.word2index[word]]
                translated_target_word.append(target_space.index2word[map_space_id])
            translated_word[word] = translated_target_word
        return translated_word

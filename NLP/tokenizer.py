import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer,TreebankWordTokenizer,SpaceTokenizer
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer


s = "Thomas Jefferson began building Monticello at the age of 26. wasn't this good?"


def statement_tokenizer(sentence):
    """
    convert statement tokenizer.
    :param sentence:
    :return:
    """
    token_seq = str.split(sentence)
    vocab = sorted(set(token_seq))
    num_tokens = len(token_seq)
    vocab_size = len(vocab)
    onehot_vector = np.zeros((num_tokens, vocab_size), int)
    for i, word in enumerate(token_seq):
        onehot_vector[i, vocab.index(word)] = 1
    one_df = pd.DataFrame(onehot_vector, columns=vocab)
    return one_df


def inbuild_tokenizer(sentence):
    """
    return tokens from statement
    :param sentence:
    :return:
    """
    # user can choose SpaceTokenizer,RegexpTokenizer etc.
    tk = TreebankWordTokenizer()
    tokens = tk.tokenize(sentence)
    return tokens


def get_ngarms(sentence,n):
    """
    return n grams from statement
    :param sentence:
    :param n:
    :return:
    """
    tokens = inbuild_tokenizer(sentence)
    ng = list(ngrams(tokens,n))
    return ng


def remove_stop_words(sentence):
    """
    stop words like a,an,the
    :param sentence:
    :return:
    """
    # define stop words
    stop_words = nltk.corpus.stopwords.words('english')
    tokens = inbuild_tokenizer(sentence)
    removed_sw_tokens = [x for x in tokens if x not in stop_words]
    return removed_sw_tokens


def apply_stemming(sentence):
    """
    apply stemming (Porter and snowball)
    :param sentence:
    :return:
    """
    stemmer = PorterStemmer()
    tokens = inbuild_tokenizer(sentence)
    stemmer_tokens = ' '.join([stemmer.stem(w) for w in tokens])
    return stemmer_tokens


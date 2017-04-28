SEED_VAL = 200;

import os
import string
import numpy as np
import pandas as pd

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix


def make_sure_path_exists(path): 
    # http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
    if not os.path.exists(path):
        os.makedirs(path)
        
def getDfInfo(df):
    nrow = df.shape[0]
#     print np.count_nonzero(df.isnull()) / nrow #http://stackoverflow.com/questions/28199524/best-way-to-count-the-number-of-rows-with-missing-values-in-a-pandas-dataframe
    print "\n*****SHAPE********"
    print df.shape
    print "*****NULL PERCENTAGE*********"
    print df.isnull().sum() / nrow

# http://stackoverflow.com/questions/11692199/string-translate-with-unicode-data-in-python
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
remove_number_map = dict((ord(char), None) for char in string.digits)

def remove_numbers_in_string(s):
#     print(type(s))
#     return s.translate(None, string.digits)
    return s.translate(remove_number_map)

def lowercase_remove_punctuation(s):
    s = s.lower()
#     s = s.translate(None, string.punctuation)
    return s.translate(remove_punctuation_map)

NLTK_STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(s):
    token_list = nltk.word_tokenize(s)
    exclude_stopwords = lambda token : token not in NLTK_STOPWORDS
    return ' '.join(filter(exclude_stopwords, token_list))

def filter_out_more_stopwords(token_list):
    return filter(lambda tok : tok not in MORE_STOPWORDS, token_list)

def stem_token_list(token_list):
    STEMMER = PorterStemmer()
#     return [STEMMER.stem(tok.decode('utf-8')) for tok in token_list]
    return [STEMMER.stem(tok) for tok in token_list]

def restring_tokens(token_list):
    return ' '.join(token_list)

def lowercase_remove_punctuation_and_numbers_and_tokenize_and_filter_more_stopwords_and_stem_and_restring(s):
    s = remove_numbers_in_string(s)
    s = lowercase_remove_punctuation(s)
    s = remove_stopwords(s)
    token_list = nltk.word_tokenize(s)
    #token_list = filter_out_more_stopwords(token_list)
    token_list = stem_token_list(token_list)
    return restring_tokens(token_list)

# http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
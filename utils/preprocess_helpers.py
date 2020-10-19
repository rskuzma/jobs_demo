### Package for cleaning and preparing text
# Richard Kuzma, 29AUG2020
# Need to make into an importable module
# This is built off of https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#3importpackages

# operations
import time
import pickle
import csv

# data
import pandas as pd

#printing
from pprint import pprint

# cleaning, lemmatizing
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger') # for part of speech tagging, required for lemmatization
nltk.download('stopwords')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# import spacy
import re

# modeling
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.test.utils import datapath
from gensim.models import CoherenceModel, TfidfModel, LdaModel


# plotting
import pyLDAvis
import pyLDAvis.gensim
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

# warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#
#
# # Remove Emails
def remove_emails(data_list):
    """takes list of strings of text, removes emails, returns list of text"""
    temp = [re.sub('\S*@\S*\s?', '', sent) for sent in data_list]
    return temp


# Remove new line characters
def remove_tabs_new_lines(data_list):
    """takes list of strings of text, removes \t and \n and replaces with ' ', returns list of text"""
    temp = [re.sub('\s+', ' ', sent) for sent in data_list]
    return temp
# data = [re.sub('\s+', ' ', sent) for sent in data]


# Remove distracting single quotes
def remove_single_quotes(data_list):
    """takes list of strings of text, removes "'", replaces with '' returns list of text"""
    temp = [re.sub("\'", "", sent) for sent in data_list]
    return temp
# data = [re.sub("\'", "", sent) for sent in data]


def sent_to_words(sentences):
    """Gensim simple preprocess. Turns each element in list of strings to a list of tokens. removes punct"""
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    common = []
    with open('./data/raw/common_words.csv', 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            common.extend(line)
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'employ', 'benefit'])
    stop_words.extend(common)
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# NLTK Lemmatize with POS Tag
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatization(texts): # texts is list of lists
    t0 = time.time()
    print('lemmatizing...')
    lemmatizer = WordNetLemmatizer()
    texts_out = []
    for doc in texts: # the document is a list
        texts_out.append([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in doc])
    print('lemmatization complete')
    print('{} seconds'.format(time.time()-t0))
    return texts_out


def make_dict(texts):
    """Takes a list of lists which contain strings (lemmatized/tokenized)"""
    """This dictionary is often referred to as id2word in NLP posts"""
    # uses gensim.corpora.Dictionary
    return corpora.Dictionary(texts)



def make_bow(texts, dictionary):
    """Creates bag of words model"""
    """texts is a list of lists; doc is a list of strings"""
    """dictionary made using make_dict"""
    # uses gensim
    return [dictionary.doc2bow(doc) for doc in texts]


# def make_tfidf(bow):
#     tfidf = TfidfModel(bow, smartirs='nfc')
#     return tfidf
#     # chars for term-freq, doc-freq, and doc-normalization weighting. Default is 'nfc'



# Human readable format of corpus (term-frequency)
def human_readable_bow(dictionary, bow_corpus, num):
    print([[(dictionary[id], freq) for id, freq in cp] for cp in bow_corpus[:num]])


def print_lda_topics(lda_model):
    pprint(lda_model.print_topics())


# Compute Perplexity
def compute_perplexity(corpus, model):
    """corpus is your bow corpus for instance"""
    """a measure of how good the model is. lower the better"""
    t0 = time.time()
    print('perplexing...')
    perplex = model.log_perplexity(corpus)
    print('{} seconds'.format(time.time()-t0))
    print('\nPerplexity: ', perplex)
    return perplex


def compute_coherence(model, texts, dictionary, coherence):
    """Takes model, texts, dictionary. Prints and outputs coherence score"""
    t0 = time.time()
    print('making coherence model...')
    coherence_model_lda = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=coherence)
    print('{} seconds'.format(time.time()-t0))
    t0 = time.time()
    print('getting coherence...')
    coherence_lda = coherence_model_lda.get_coherence()
    print('{} seconds'.format(time.time()-t0))
    print('\nCoherence Score: ', coherence_lda)
    return coherence_lda


def visualize_lda(model, corpus, dictionary):
    """returns the pyLDAvis PreparedData given model, corpus, dictionary"""
    """Could pickle this to save it"""
    """pyLDAvis.save_html(vis, "filename") also works to export in html"""
    pyLDAvis.enable_notebook()
    t0 = time.time()
    vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    print('{} seconds'.format(time.time()-t0))
    return vis


def full_clean(data):
    """data is a list of strings, uncleaned, each element a document"""

    # remove emails
    data = remove_emails(data)

    # remove tabs and new lines
    data = remove_tabs_new_lines(data)

    # remove single quotes
    data = remove_single_quotes(data)

    # tokenize sentences into list of words
    data_words = list(sent_to_words(data))


    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    ### Build bigram model
    bigram = gensim.models.Phrases(data_words_nostops, min_count=5, threshold=100) # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]

    ### Build trigram model
    trigram = gensim.models.Phrases(bigram[data_words_nostops], min_count=5, threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    data_words_trigrams = [trigram_mod[bigram_mod[doc]] for doc in data_words_bigrams]

    # return a lemmatized list of unigram + bigram + trigrams
    return lemmatization(data_words_trigrams)

def compute_compare_c_v(dictionary, corpus, texts, rg=[2, 20, 6]):
    """
    Compute and compare c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    perplex_values = []
    model_list = []
    print('range entered: {}'.format(rg))
    for num_topics in rg:
        print('num_topics: {}'.format(num_topics))

        print('\nmaking model with {} topics...'.format(num_topics))

        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        print('model made: {}'.format(model))
        print('old model_list: {}'.format(model_list))
        model_list.append(model)
        print('appended. model_list now: {}'.format(model_list))

        print('\nperplexity...')
        p_score = compute_perplexity(corpus, model)
        perplex_values.append(p_score)
        print('appended perplexity')

        print('\ncoherence...')
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        c_score = compute_coherence(model, texts, dictionary, 'c_v') # my coherence model line
#         print('coherence: {}'.format(coherencemodel.get_coherence()))
#         coherence_values.append(coherencemodel.get_coherence())
        coherence_values.append(c_score)
        print('appended coherence')


    return model_list, coherence_values, perplex_values


if __name__ == '__main__':
    print('if __name__ == \'__main__\': triggered...')
    print('Runnning preprocess_helpers.py instead of importing...')
    pass

import os
import sys
import csv
import math
import random
import gensim as gs
from nltk import tokenize
import pandas as pd
import numpy as np
import tensorflow as tf
import zipfile
import collections
from six.moves import xrange
import multiprocessing
from gensim.corpora.wikicorpus import WikiCorpus

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from tfword2vec import TFWord2Vec


class WordEmbeddings:
    def __init__(self):
        self.test1=0

    ######################################################################################################

    def debugPrintRandomSentences(self, sentences, n):
        """
        Print n random sentences taken from 'sentences'
        :param sentences: the list of sentences making up the text
        :param n: number of random samples to print i.e. 10
        :return:
        """
        print('Random example sentences:')
        for i in range(n):
            print(i, random.choice(sentences))

    ######################################################################################################

    def batchWordsToSentences(self, words, n):
        """
        Take a list of individual words and batch them into sentences of 'n' words each.
        Not an intelligent words to sentences, but designed to take a file with just words and build
        sentence lists that are usable by gensim.
        NOTE: an alternative technique (used by gensim) is to read chunks of characters and split them into sentences of
        variable numbers of words. You could also split randomly using this method and include an upper and lower bound
        on sentence length i.e. between 10 and 20 words.
        :param words: list of individual words
        :param n: number of words to put into the batch e.g. 10
        :return: a list containing lists of n words
        """
        sentences = []
        count = 0
        sentence = []
        for word in words:
            count = count + 1
            if count == n:
                sentences.append(sentence)
                count = 0
                sentence = []
            sentence.append(word)
        if len(sentence) > 0:
            sentences.append(sentence)
        return sentences

    ######################################################################################################

    def makeWikiTextEmbedding(self):
        # on wiki text sentences
        wiki = WikiCorpus('data/swwiki-latest-pages-articles.xml.bz2', lemmatize=False, dictionary={})
        sentences = list(wiki.get_texts())
        print("wikitext: ",len(sentences)," sentences")
        self.debugPrintRandomSentences(sentences,10)
        model = gs.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=8, sg=1, hs=1, iter=15)

    def makeText8Embedding(self):
        # gensim text 8 - two versions, mine and the gensim text 8 corpus reader, which uses an iterator
        # on the text8 file, which we have to batch up into sentences in a very naive way
        # with open('data/text8/text8', 'rt') as f:
        #    words = f.read().split()
        # text8 = batchWordsToSentences(words,10)
        # debugPrintRandomSentences(text8, 10)
        text8 = gs.models.word2vec.Text8Corpus(
            'data/text8/text8')  # NOTE: this splits the one line text 8 into 8192 character blocks of words
        model = gs.models.Word2Vec(text8, size=100, window=5, min_count=5, workers=8, sg=1, hs=1, iter=5)
        return model

    ######################################################################################################
    # PLOTTING for gensim and TF word2vec

    # pylint: disable=missing-docstring
    # Function to draw visualization of distance between embeddings.
    def plot_with_labels(low_dim_embs, labels, filename):
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(36, 36))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.savefig(filename)

    # def plottSNE(model):
    #    try:
    #        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    #        plot_only = 500
    #        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    #        labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    #        #plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))
    #        plot_with_labels(low_dim_embs, labels, 'tsne.png')
    #    except ImportError as ex:
    #        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    #        print(ex)


    ######################################################################################################


    def gensimWord2Vec(self, texts):
        """
        gensim word2vec, which I don't think works nearly as fast or as well as the tensorflow version
        :param texts: list of corpus filenames to use
        :return:
        """
        # gensim
        sentences = []
        for filename in texts:
            print('Parsing file', filename)
            with open(filename, 'rt') as f:
                text = f.read()
            print("Plain text size: ", len(text))
            # take out text crlf characters so it's just one long string
            text = text.replace('\n', ' ')
            text = text.replace('\r', ' ')
            newsentences = tokenize.sent_tokenize(text)  # use nltk to convert plain text into sentences
            newsentences = [s.split() for s in newsentences]  # split sentences into lists of words
            # todo: if these are sentences, then you can take the capital off of just the first word and full-stop off the last
            for newsen in newsentences:
                # all word filtering
                for i in range(0, len(newsen)):
                    word = newsen[i]
                    word = word.replace('"', '')
                    # word = word.replace('\'', '') #NO: I've etc
                    word = word.replace(',', '')
                    word = word.replace(';', '')
                    word = word.replace(':', '')
                    newsen[i] = word
                # first word filtering
                newsen[0] = newsen[0].lower()  # fails if the first word is a proper noun
                # final word filtering
                lastword = newsen[-1]
                if lastword.endswith('.') or lastword.endswith('!'):
                    lastword = lastword[:-1]
                newsen[-1] = lastword

            print(len(newsentences), "sentences parsed.")
            sentences.extend(newsentences)

        print('Total sentences: ', len(sentences))
        print('Random example sentences:')
        for i in range(10):
            print(i, random.choice(sentences))
        # sg=0 CBOW, sg=1 Skip gram, hs=1 use softmax, iter=number of iterations (epochs)
        # window is number of words in skipgram, min_count drops words less than this as too infrequent for analysis
        model = gs.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=8, sg=1, hs=1, iter=15)
        # print("model computer: ",model["computer"])
        # model.save(fname)
        # model = Word2Vec.load(fname)  # you can continue training with the loaded model!

        # plot word embeddings using t-SNE view in 2D
        vocab = list(model.wv.vocab)  # list of words in the vocabulary so we can get them in the right order
        embeds = model[vocab]  # list of word embedding vectors
        # check they're in the same order
        # print('check embed ordering is correct')
        # print(labels[0],'=',embeds[0])
        # print(labels[0],'=',model[labels[0]])
        plot_only = 500
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        low_dim_embeds = tsne.fit_transform(embeds[:plot_only, :])
        labels = [vocab[i] for i in range(0, plot_only)]
        # sanity check that they're in sync - well, at least they were in sync when labels and embeds went into tSNE
        # print(labels[0],low_dim_embeds[0])
        # print(labels[0],embeds[0])
        # print(labels[0],model[labels[0]])
        self.plot_with_labels(low_dim_embeds,labels,'tsne_gensim.png')
        #end of plotting
        return model

###############################################################################
#TENSORFLOW

    def tfWord2Vec(self, filename):
        """
        Input is a corpus of words (not sentences) in a file
        :param filename:
        :return:
        """

        with zipfile.ZipFile(filename) as f:
            data = tf.compat.as_str(f.read(f.namelist()[0])).split()
            tfword2vec = TFWord2Vec()
            embeds = tfword2vec.word2vec(data)

        #and visualise!
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(tfword2vec.final_embeddings[:plot_only, :])
        labels = [tfword2vec.reverse_dictionary[i] for i in xrange(plot_only)]
        self.plot_with_labels(low_dim_embs, labels, 'tsne_tf.png')

        return tfword2vec

        ###############################################################################
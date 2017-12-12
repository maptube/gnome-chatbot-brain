#!/usr/bin/env python3
#This is a super gnome implementation from Gnome Industries
#TODO: look at gensim for word2vec in addition to tf


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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from tfword2vec import TFWord2Vec

creatureNames = ['Parker', 'Wombat', 'Loki', 'Gnomeo', 'Jetpack Gnomey', 'Super Gnome', 'Yusuf',
                     'Denchu', 'Zack', 'Khadija',
                     'Rosie', 'Beehigh',
                     'Moonlight', 'Goku', 'Shadow Blade']

# chatbot text following the user's yes or no after memory prompt
# TODO: could also track "username YES|NO"
#memoryTextYes = [
#        "Perfect! What would you like to tell me?",
#        "Great! What would you like to tell me?",
#        "Great! What's your memory about?"]
#memoryTextNo = "No problem "  # plus name

# chatbot text when user meets a second gnome
#otherGnomeText = [
#        "I see you've already spoken to ",  # plus creature name
#        "You just spoke to ",  # plus creature name
#        "I see you've just spoken to "  # plus creature name
#    ]

# nextCreatureText = "You can find my buddy "


#list of conversations
conversations = []

class ConversationLine:
    """One line of the conversation"""
    def __init__(self, speaker, text):
        self.speaker = speaker
        self.text = text;


############################################################################################################

def loadGnomeData(infile):
    """
    Load gnome conversations from csv file into conversations structure (list of ConversationLine list)
    Precondition: infile conversation list must be grouped by cid number
    :param infile:
    :return:
    """
    with open(infile, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader) #skip header
        lastcid = -1
        conversation = [] #ordered list of ConversationLine objects
        for row in reader:
            cid = row[0]
            name = row[1]
            text = row[2]
            if cid!=lastcid and len(conversation)>0:
                #different conversation id and not the first
                conversations.append(conversation)
                conversation=[] #start a new conversation
            lastcid=cid
            conversation.append(ConversationLine(name,text))
        #don't forget to push the last conversation in the file...
        conversations.append(conversation)

######################################################################################################

def wordCountFromGnomeData():
    """
    Compute a word count from the gnome data loaded into the conversations array.
    PRE: most have called loadGnomeData first to load the data
    :return:
    """
    wordcount = {}
    for conversation in conversations:
        for convline in conversation:
            words = convline.text.split(' ')
            for word in words:
                word = word.lower()
                word = word.replace(',', '')
                word = word.replace('.', '')
                # word = word.replace('\'','')
                word = word.replace('?', '')
                word = word.replace('!', '')
                if word in wordcount:
                    wordcount[word] = wordcount[word] + 1
                else:
                    wordcount[word] = 1

    return wordcount

######################################################################################################

def wordCountFromPlainTextFile(infile):
    """
    Read words from a plain text file and return a map of [word,count]
    :param infile:
    :return:
    """
    wordcount = {}
    with open(infile, 'rt') as textfile:
        line = textfile.readline()
        words = line.split(' ')
        for word in words:
            word = word.lower()
            word = word.replace(',', '')
            word = word.replace('.', '')
            # word = word.replace('\'','')
            word = word.replace('?', '')
            word = word.replace('!', '')
            if word in wordcount:
                wordcount[word] = wordcount[word] + 1
            else:
                wordcount[word] = 1

    return wordcount

######################################################################################################

def compute_unigramProbabilities(wordcount):
    """
    Compute word probabilities from a word count hash. In other words just normalise by the sum and return.
    :return: hash of [word,probability]
    """
    #NOTE: this is a terrible implementation of unigram probs - much too naive


    #now convert counts to probabilities
    sum = 0
    for key, value in wordcount.items():
        sum = sum+value

    #and normalise
    for key, value in wordcount.items():
        wordcount[key] = value/sum

    return wordcount

######################################################################################################

#skipgrams


######################################################################################################

def makeWordFile(infiles,outfile):
    """
    Turn a block of plain text book words into a word file that we can use with word2vec
    :param infiles: list of input filenames
    :param outfilename:
    :return:
    """
    count=0
    with open(outfile,'wt') as wordfile:
        for filename in infiles:
            with open(filename, 'rt') as textfile:
                for line in textfile:
                    line = line.rstrip('\n')
                    line = line.rstrip('\r')
                    words = line.split(' ')
                    for word in words: #I know, WHITELIST
                        word = word.strip()
                        word = word.lower()
                        word = word.replace(',', '')
                        word = word.replace('.', '')
                        word = word.replace('\'','')
                        word = word.replace('\\', '')
                        word = word.replace('/', '')
                        word = word.replace('"','')
                        word = word.replace('?', '')
                        word = word.replace('!', '')
                        word = word.replace('(', '')
                        word = word.replace(')', '')
                        word = word.replace(':', '')
                        word = word.replace(';', '')
                        word = word.replace('{', '')
                        word = word.replace('}', '')
                        #word = word.replace('-','')
                        word = word.replace('|', '')
                        word = word.replace('*', '')
                        word = word.replace('`', '')
                        word = word.replace('>', '')
                        word = word.replace('<', '')
                        word = word.replace('=', '')
                        word = word.strip()
                        if len(word)>0 and word!='-':
                            wordfile.write(word+' ')
                            count=count+1
    print("word count=",count)

######################################################################################################

#def plainTextToSentences(text):
#    """
#    Use nltk to convert a block of plain text (i.e. a book) into sentences
#    :param text: Plain text unformatted input.
#    :return: list of sentences
#    """
#    return tokenize.sent_tokenize(p)

######################################################################################################
#PLOTTING for gensim and TF word2vec

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

#def plottSNE(model):
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

def gensimWord2Vec(texts):
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
    plot_with_labels(low_dim_embeds,labels,'tsne_gensim.png')
    #end of plotting
    return model

###############################################################################
#TENSORFLOW

def tfWord2Vec(filename):
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
    plot_with_labels(low_dim_embs, labels, 'tsne_tf.png')

    return tfword2vec

###############################################################################

def main():
    #infilename = "C:\\Users\\richard\\Desktop\\Gnomes\\SIGCHI\\gnome-data\\conversations.csv"
    #infilename = "C:\\Users\\richard\\Desktop\\SIGCHI\\gnomes-data\\20170911_conversations\\conversations_sep.csv"
    #infilename = "C:\\Users\\richard\\Dropbox\\SIGCHI\\conversations.csv"
    infilename = "C:\\Users\\richard\\Desktop\\gnomes-data\\20171101_conversations\\conversations_sep.csv"
    ##
    corpus_hhgttg = "data\\hhgttg.txt"
    corpus_fish = "data\\fish.txt"
    corpus_life = "data\\life.txt"
    corpus_rest = "data\\rest.txt"
    corpus_harmless = "data\\harmless.txt"

    #loadData(infilename)
    #compute_unigramProbabilities()

    #makeWordFile([corpus_hhgttg, corpus_fish, corpus_life, corpus_rest, corpus_harmless],'hhgttg_words.txt')
    #wordcounts = wordCountFromPlainTextFile('hhgttg_words.txt')
    #print("distinct words = ",len(wordcounts.keys()))

    ##

    #gensim model

    #model = gensimWord2Vec([corpus_hhgttg, corpus_fish, corpus_life, corpus_rest, corpus_harmless])
    #now some similarity checks
    #print("woman, man: ",model.wv.similarity('woman', 'man'))
    #print("life, universe: ", model.wv.similarity('life', 'universe'))
    #print("life, everything: ", model.wv.similarity('life', 'everything'))
    #print("universe, everything: ", model.wv.similarity('universe', 'everything'))

    ##

    #Tensorflow word2vec
    tfword2vec = tfWord2Vec('text8.zip')
    #tfword2vec = tfWord2Vec('data/hhgttg_words.zip')
    #now some similarity checks
    print("woman, man: ",tfword2vec.cosine_similarity('woman','man'))
    print("life, universe: ", tfword2vec.cosine_similarity('life', 'universe'))
    print("life, everything: ", tfword2vec.cosine_similarity('life', 'everything'))
    print("universe, everything: ", tfword2vec.cosine_similarity('universe', 'everything'))
    print("england, london: ", tfword2vec.cosine_similarity('england', 'london'))
    print("france, paris: ", tfword2vec.cosine_similarity('france', 'paris'))
    print("greece, athens: ", tfword2vec.cosine_similarity('greece', 'athens'))


###############################################################################
if __name__ == "__main__":
    main()

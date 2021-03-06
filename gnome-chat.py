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
import multiprocessing
#from gensim.corpora.wikicorpus import WikiCorpus
import gensim as gs

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from tfword2vec import TFWord2Vec
from wordembeddings import WordEmbeddings
#from seq2seq import Seq2Seq
import seq2seq as s2s


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
    def __init__(self, cid, speaker, text):
        self.cid = cid
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
    conversations.clear()
    with open(infile, 'rt', encoding='utf-8') as csvfile:
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
            conversation.append(ConversationLine(cid,name,text))
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

#def filterWord(w):


######################################################################################################
def conditionText(text):
    #Make text lowercase
    #Quote and comma give problems with loading and saving words, so make them into special symbols.
    #For other symbols, just make sure they have separating spaces
    #TODO: a better way of doing this might be to split into words, then you could filter this: 'single quoted text'
    text = text.encode("ascii", "replace").decode("utf-8") # I want plain old ascii, not utf-8
    text = text.lower()
    text = text.replace("\"", " <QUOTE> ")
    text = text.replace(",", " <COMMA> ")
    text = text.replace("!", " ! ")
    text = text.replace("?", " ? ")
    text = text.replace(".", " . ")
    text = text.replace(":", " : ")
    text = text.replace(";", " : ")
    #text = text.replace('`', '\'') #solves an encoding problem
    text = text.replace("  ", " ") #now that is a real hack! replace any double spaces we just introduced with a single one!
    text = text.replace("  ", " ") #and again!
    return text

def conditionGnomeChat(inFilename, outFilename):
    """
    Read in the gnome chat data that we collected (csv file) and generate gnome/visitor pairs of sentences.
    The raw data splits gnome lines over multiple lines, which we stick back together here and add <SOC> <EOC>
    start and end of conversation markers. Inside this we also add <SOT> and <EOT> markers for each speaker's
    start and end of text.
    :param inFilename:
    :param outFilename:
    :return:
    """
    loadGnomeData(inFilename)
    with open(outFilename,'wt') as outfile:
        for conv in conversations:
            #outfile.write("<SOC>\n") #start of conversation
            lastSpeaker = ""
            currentText=""
            for cvl in conv:
                if (cvl.speaker!=lastSpeaker):
                    if len(currentText)>0:
                        #outfile.write("<SOT> "+currentText.strip()+" <EOT> \n") #Start of text and end of text
                        outfile.write(currentText.strip() + " <EOT> \n")  # Start of text and end of text
                        currentText=""
                currentText = currentText+" "+conditionText(cvl.text)
                lastSpeaker=cvl.speaker
            #don't forget to write out the last line if it's still in currentText
            if len(currentText)>0:
                #outfile.write("<SOT> " + currentText.strip() + " <EOT> \n")  # Start of text and end of text
                outfile.write(currentText.strip() + " <EOT> \n")  # Start of text and end of text
            #outfile.write("<EOC>\n") #end of conversation

######################################################################################################

def makeWordFile(infiles,outfile):
    """
    Turn a block of plain text book words into a word file that we can use with word2vec
    :param infiles: list of input filenames
    :param outfilename:
    :return:
    """
    #NOTE: you can do this: count = collections.Counter(words).most_common()
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

def computeSentenceLengths(inFilename,outFilename):
    """
    Processing for beginning half of sentence length structure against end sentence length structure.
    Only acts on visitor conversations.
    This is a test for whether the sentence structure breaks down as the conversation goes on i.e. as they realise
    they're talking to a mechanism, the language gets simpler. If, however, they leave a memory at the end, then the
    memories tend to be longer (as they're being told), so the second half of the conversation will have more words.
    That's the hypothesis anyway...
    :param inFilename: location of the gnome text corpus
    :param outFilename: file to write out containing the data
    :return:
    """
    loadGnomeData(inFilename)
    print("computeSentenceLengths::loaded ",len(conversations)," conversations")
    with open(outFilename, 'wt', encoding='utf-8') as outfile:
        outfile.write("cid,speaker,lines,totalwords,firsthalf,secondhalf,bagofwords\n")
        for conv in conversations:
            wordlens = []
            wordbag = []
            lines = 0 #number of visitor conversation lines
            totalWords = 0 #total number of words spoken by visitor in this conversation
            speaker = ''
            cid = 0
            for cvl in conv:
                if cvl.speaker not in creatureNames:
                    speaker=cvl.speaker
                    cid=cvl.cid
                    lines=lines+1
                    words = cvl.text.split()
                    wordlens.append(len(words))
                    wordbag=wordbag + words
                    totalWords=totalWords+len(words)
                    firstHalf=0 #total words in first half of conversation
                    secondHalf=0 #total words in second half of conversation
                    i=0
                    j=len(wordlens)-1
                    while (i<=j):
                        if i==j: #edge case, if conv lines are odd, then split the middle conversation length in half
                            firstHalf = firstHalf + wordlens[i]/2
                            secondHalf = secondHalf + wordlens[j]/2
                        else:
                            firstHalf=firstHalf+wordlens[i]
                            secondHalf=secondHalf+wordlens[j]
                        i=i+1
                        j=j-1

            outfile.write(cid+","+speaker+","+str(lines)+","+str(totalWords)+","+str(firstHalf)+","+str(secondHalf)+","+" ".join(wordbag)+"\n")

######################################################################################################

def maxCloseness(model,patternWords,words):
    """
    Test closeness of "words" to "patternWords" i.e. given pattern=["hello", "gnome"] and words=["hello", "there", "gnome", ",", "my", "name" "is"]
    then you should get a closeness match on hello and gnome, so it returns the maximum.
    :param model: The text model containing the word embedding vectors to use.
    :param patternWords: The pattern that we're matching against as a list of words i.e. ["hello", "hi", "hiya"] to match a hello context.
    :param words: List of plain text words that we're matching against.
    :return: The minimum closeness value of the pattern words applied over the text.
    TODO: isn't closeness commutative? i.e. the nested loops should be triangluar?
    """
    cosmax=-1 #it's cosine closeness, 1 means close (zero angle)
    cosword="NONE"
    for word1 in patternWords:
        word1 = word1.lower()
        for word2 in words:
            word2 = word2.lower()
            #need to check word is in the vocabulary, otherwise you get an error
            if word1 in model.wv.vocab and word2 in model.wv.vocab and word2 != 'hello':
                val = model.wv.similarity(word1,word2)
                if (val>cosmax):
                    cosmax=val
                    cosword=word2
    return cosmax, cosword

######################################################################################################


def textProbability(modelText8, modelPark, inFilename, outFilename):
    """
    Compute word probabilities based on various text models
    :param inFilename:
    :param outFilename:
    :return:
    """
    loadGnomeData(inFilename)
    print("textProbability::loaded ", len(conversations), " conversations")
    #embeds = WordEmbeddings()
    #modelText8 = embeds.makeText8Embedding()
    #modelText8 = gs.models.Word2Vec.load('lm/gensim-text8.model')
    #print("text8 score: hello there my name is fred")
    #score = modelText8.score([["hello", "there", "everybody"]])
    #print("score=",score)
    #print("closeness: ",modelText8.wv.similarity("i","am"))
    with open(outFilename, 'wt', encoding='utf-8') as outfile:
        outfile.write("cid,speaker,lines,totalwords,probText8,probPark,probPersonal,personalWord,probMonth,monthWord,bagofwords\n")
        for conv in conversations:
            wordlens = []
            wordbag = []
            lines = 0  # number of visitor conversation lines
            totalWords = 0  # total number of words spoken by visitor in this conversation
            speaker = ''
            cid = 0
            for cvl in conv:
                if cvl.speaker not in creatureNames:
                    speaker = cvl.speaker
                    cid = cvl.cid
                    lines = lines + 1
                    words = cvl.text.split()
                    wordlens.append(len(words))
                    wordbag = wordbag + words
                    totalWords = totalWords + len(words)
                    probText8 = modelText8.score([words]) #yes, it's a list of bags of words
                    probPark = modelPark.score([words])
                    probPersonal, personalWord = maxCloseness(modelText8,["i","he","she","it","they","we"],words)
                    probMonth, monthWord = maxCloseness(modelText8,
                            ["january","february","march","april","may","june","july","august","september","october","november","december"],words)
            outfile.write(
                cid + "," + speaker + ","
                + str(lines) + "," + str(totalWords) + ","
                + str(probText8[0]) + ","
                + str(probPark[0]) + ","
                + str(probPersonal) + "," + personalWord + ","
                + str(probMonth) + "," + monthWord + ","
                + " ".join(wordbag)
                + "\n")


######################################################################################################

def trainWordEmbeddings(outFilename):
    """train a word vector embedding which we save for later"""

    corpus_hhgttg = "data\\hhgttg.txt"
    corpus_fish = "data\\fish.txt"
    corpus_life = "data\\life.txt"
    corpus_rest = "data\\rest.txt"
    corpus_harmless = "data\\harmless.txt"

    # loadData(infilename)
    # compute_unigramProbabilities()

    # makeWordFile([corpus_hhgttg, corpus_fish, corpus_life, corpus_rest, corpus_harmless],'hhgttg_words.txt')
    # wordcounts = wordCountFromPlainTextFile('hhgttg_words.txt')
    # print("distinct words = ",len(wordcounts.keys()))

    we = WordEmbeddings()
    model = we.makeText8Embedding() #using gensim
    model.save(outFilename)

    #other versions here in case we want to do a test
    # on hhgttg
    # model = gensimWord2Vec([corpus_hhgttg, corpus_fish, corpus_life, corpus_rest, corpus_harmless])
    # Tensorflow word2vec
    # tfword2vec = tfWord2Vec('data/text8.zip')
    # tfword2vec = tfWord2Vec('data/hhgttg_words.zip')

    # now some similarity checks
    print("woman, man: ", model.wv.similarity('woman', 'man'))
    print("life, universe: ", model.wv.similarity('life', 'universe'))
    print("life, everything: ", model.wv.similarity('life', 'everything'))
    print("universe, everything: ", model.wv.similarity('universe', 'everything'))
    print("england, london: ", model.wv.similarity('england', 'london'))
    print("france, paris: ", model.wv.similarity('france', 'paris'))
    print("greece, athens: ", model.wv.similarity('greece', 'athens'))
    print("would, will: ", model.wv.similarity('would', 'will'))

    return model

def trainChatbot():
    """Train a chatbot using a word embedding and a file containing [context, "pattern"] tuples"""
    #model is a gensim word2vec model with 100 vector words
    #inFile:
    #   NAME, "my name is $NAME$"

    #load the training data
    #data = dict()
    #with open(inFilename, 'rt') as csvfile:
    #    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    #    for row in csvreader:
    #        ctx = row[0]
    #        pattern = row[1]
    #        print("trainChatbot:",ctx,pattern)
    #        if ctx in data:
    #            data[ctx].append(pattern)
    #        else:
    #            data[ctx]=[pattern]

    #print("Found ",len(data.keys())," chatbot contexts")

    #print("Begin training")
    #seq2seq = s2s.Seq2Seq()
    text = ('long ago , the mice had a general council to consider what measures they could take to outwit their '
    'common enemy , the cat . some said this , and some said that but at last a young mouse got up and said he had '
    'a proposal to make , which he thought would meet the case . you will all agree , said he , that our chief danger '
    'consists in the sly and treacherous manner in which the enemy approaches us . now , if we could receive some '
    'signal of her approach , we could easily escape from her . i venture , therefore , to propose that a small bell '
    'be procured , and attached by a ribbon round the neck of the cat . by this means we should always know when she '
    'was about , and could easily retire while she was in the neighbourhood . this proposal met with general applause '
    ', until an old mouse got up and said that is all very well , but who is to bell the cat ? the mice looked at one '
    'another and nobody spoke . then the old mouse said it is easy to propose impossible remedies .')
    #seq2seq.trainLMTest(text.split()) #note words passed in are an array of words
    #s2s.test_trainLMEmbedding(model, text.split())

    #train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()
    #training data needs a list of words as word ids, for which we need a dictionary of words
    words = s2s.readWords("lm\\clean_gnomechat.txt") #returns list of individual words from the cleaned chat file
    dictionary, reversed_dictionary = s2s.buildDictionary(words) #returns word->id and id->word dictionaries
    s2s.saveWordDictionaryTSV(reversed_dictionary,"lm\\rdictionary.csv")
    word_ids = s2s.wordsToWordIds(words,dictionary) #having got a word->id lookup, convert all the words
    print("Loaded word file: ",len(words), " words in chat file, ",len(dictionary)," words in dictionary")
    print("Begin training")
    s2s.trainLMAdvanced(word_ids, len(dictionary), num_layers=2, num_epochs=60, batch_size=20, print_iter=10,
          model_save_name='two-layer-lstm-medium-config-60-epoch-0p93-lr-decay-10-max-lr')

def testChatbot():
    #don't strictly need the corpus text, except for the fact that it's in the Seq2Seq constructor - need to remove this.
    words = s2s.readWords("lm\\clean_gnomechat.txt")  # returns list of individual words from the cleaned chat file
    #inputText = "loki ! ! speak to me !  loki ! ! speak to me !  hello <EOT>"
    #words=[]
    #for i in range(0,1000):
    #    words = words + inputText.split()
    print("word length = ", len(words))
    model_path="rnn_words\\two-layer-lstm-medium-config-60-epoch-0p93-lr-decay-10-max-lr-final"
    reversed_dictionary = s2s.loadWordDictionaryTSV("lm\\rdictionary.csv")
    print("reversed_dictionary words=",len(reversed_dictionary.keys()))
    dictionary = dict([[v, k] for k, v in reversed_dictionary.items()]) #reversed reverse dictionary
    word_ids = s2s.wordsToWordIds(words, dictionary)  # having got a word->id lookup, convert all the words
    print("words ids=",word_ids)
    s2s.testLMAdvanced(model_path, word_ids, dictionary, reversed_dictionary)



###############################################################################

def main():
    #infilename = "C:\\Users\\richard\\Desktop\\Gnomes\\SIGCHI\\gnome-data\\conversations.csv"
    #infilename = "C:\\Users\\richard\\Desktop\\SIGCHI\\gnomes-data\\20170911_conversations\\conversations_sep.csv"
    #infilename = "C:\\Users\\richard\\Dropbox\\SIGCHI\\conversations.csv"
    #infilename = "C:\\Users\\richard\\Desktop\\gnomes-data\\20171101_conversations\\conversations_sep.csv"
    #infilename = "gnomes-data\\20171101_conversations\\conversations_sep.csv"
    #infilename = "gnomes-data\\20171214_conversations\\conversations_sep.csv"
    infilename = "C:\\Users\\richard\\Desktop\\gnomes-data\\20171214_conversations\\conversations_sep.csv"
    ##

    #only run once code to create the word embedding vector files that we're using here
    #directories are hard coded in to be lm/filename.model
    #we = WordEmbeddings()
    #text8 = we.makeText8Embedding()
    #text8.save('lm/gensim-text8.model')
    #park = we.makeParkTextEmbedding()
    #park.save('lm/gensim-park.model')

    #load the gnome data and write out all user response lines as plain text
    loadGnomeData(infilename)
    for conv in conversations:
        for cvl in conv:
            if not cvl.speaker in creatureNames:
                words = cvl.text.split()
                print(cvl.speaker,",",len(words),",\"",cvl.text,"\"",) #print sentence and number of words
    #        #print(cvl.speaker, ",\"", cvl.text, "\"")

    conditionGnomeChat(infilename,"lm/clean_gnomechat.txt")

    computeSentenceLengths(infilename, "lm\\firstHalfSecondHalf.csv")

    # modelText8 = embeds.makeText8Embedding()
    modelText8 = gs.models.Word2Vec.load('lm/gensim-text8.model')
    modelPark = gs.models.Word2Vec.load('lm/gensim-park.model')
    textProbability(modelText8, modelPark, infilename, "lm\\gensim-text8-textProbability.csv")

    ##

    #Train the language model on words - this is using the text8 corpus
    #model = trainWordEmbeddings('lm/gensim-text8.model')
    #we = WordEmbeddings()
    #model = we.gensimWord2Vec(['lm/clean_gnomechat.txt'])
    #print("model generated")
    #model = gs.models.Word2Vec.load('lm/gensim-text8.model')
    #print("arcelormittal, ?: ", model.wv.most_similar_cosmul(positive=['arcelormittal']))
    #print("place, ?: ", model.wv.most_similar_cosmul(positive=['place']))
    #print("duck, ?: ", model.wv.most_similar_cosmul(positive=['duck']))
    #model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
    #print("score", model.score(['hello gnome'.split()]))


    #Training
    #model = gs.models.Word2Vec.load('lm/gensim-text8.model')
    trainChatbot()
    #testChatbot()




###############################################################################
if __name__ == "__main__":
    main()

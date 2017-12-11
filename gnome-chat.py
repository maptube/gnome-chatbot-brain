#!/usr/bin/env python3
#This is a super gnome implementation from Gnome Industries
#TODO: look at gensim for word2vec in addition to tf


import os
import sys
import csv
import random
import gensim as gs
from nltk import tokenize


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

def main():
    #infilename = "C:\\Users\\richard\\Desktop\\Gnomes\\SIGCHI\\gnome-data\\conversations.csv"
    #infilename = "C:\\Users\\richard\\Desktop\\SIGCHI\\gnomes-data\\20170911_conversations\\conversations_sep.csv"
    #infilename = "C:\\Users\\richard\\Dropbox\\SIGCHI\\conversations.csv"
    infilename = "C:\\Users\\richard\\Desktop\\gnomes-data\\20171101_conversations\\conversations_sep.csv"
    ##
    corpus_hhgttg = "D:\\richard\\GitHub\\gnome-chatbot-custom\\data\\hhgttg.txt"
    corpus_fish = "D:\\richard\\GitHub\\gnome-chatbot-custom\\data\\fish.txt"
    corpus_life = "D:\\richard\\GitHub\\gnome-chatbot-custom\\data\\life.txt"
    corpus_rest = "D:\\richard\\GitHub\\gnome-chatbot-custom\\data\\rest.txt"
    corpus_harmless = "D:\\richard\\GitHub\\gnome-chatbot-custom\\data\\harmless.txt"

    #loadData(infilename)
    #compute_unigramProbabilities()

    #makeWordFile([corpus_hhgttg, corpus_fish, corpus_life, corpus_rest, corpus_harmless],'hhgttg_words.txt')
    #wordcounts = wordCountFromPlainTextFile('hhgttg_words.txt')
    #print("distinct words = ",len(wordcounts.keys()))

    #gensim
    sentences = []
    for filename in [corpus_hhgttg, corpus_fish, corpus_life, corpus_rest, corpus_harmless]:
        print('Parsing file',filename)
        with open(filename,'rt') as f:
            text = f.read()
        print("Plain text size: ",len(text))
        #take out text crlf characters so it's just one lone string
        text=text.replace('\n',' ')
        text=text.replace('\r',' ')
        newsentences = tokenize.sent_tokenize(text) #use nltk to convert plain text into sentences
        newsentences = [s.split() for s in newsentences] #split sentences into lists of words
        #todo: if these are sentences, then you can take the capital off of just the first word
        print(len(newsentences),"sentences parsed.")
        sentences.extend(newsentences)

    print('Total sentences: ',len(sentences))
    print('Random example sentences:')
    for i in range(10):
        print(i,random.choice(sentences))
    model = gs.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    print("model computer: ",model["computer"])
    #model.save(fname)
    #model = Word2Vec.load(fname)  # you can continue training with the loaded model!


###############################################################################
if __name__ == "__main__":
    main()

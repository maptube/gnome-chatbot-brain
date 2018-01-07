
import random
import time
import collections
import datetime as dt
import csv

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


#this is good on contextual chatbots with tf
#https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077

#simpler LSTM example: https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/RNN/rnn_words.py
#or read this
#http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/

######################################################################################################

def readWords(infilename):
    """
    Read in the cleaned text file which already contains <soc> <eoc> <sot> <eot> tags
    :param infilename:
    :return: data from the input file split into a list of words
    """
    with tf.gfile.GFile(infilename, "r") as f:
        text = f.read()
        # condition here
        # return f.read().decode("utf-8").replace("\n", "<eos>").split()
        return text.split()


######################################################################################################

def buildDictionary(words):
    """
    Build a dictionary containing word->id and a reverse_dictionary containing id->word
    The dictionary is built in word frequency order with the most frequent word having the lowest word id.
    That way it's easy to chop off the back end if we want to reduce the vocabulary.
    This is basically from the Tensorflow example.
    :param infilename:
    :return: dictionary and reverse_dictionary
    """

    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    dictionary = dict(zip(words, range(len(words))))
    reverse_dictionary = dict([[v, k] for k, v in dictionary.items()])

    return dictionary, reverse_dictionary

######################################################################################################

def saveWordDictionaryTSV(reverse_dictionary,outFilename):
    """
    Save a word dictionary as a tsv file (tab separated), which is what the tensorboard needs.
    NOTE: the row ordering MUST match the word id (i.e. it's in ascending id order) to match the tensor. First line
    is a header line.
    :param reverse_dictionary: the wordid to word string lookup
    :param outFilename: filename to write data out to
    :return:
    """
    with open(outFilename, 'wt') as f:
        f.write("keyword\twordid\n")
        for i in range(0,len(reverse_dictionary.keys())):
            k = reverse_dictionary[i]
            k2 = k.encode("ascii","replace").decode("utf-8")
            f.write(""+k2+"\t"+str(i)+"\n")


######################################################################################################

def loadWordDictionaryTSV(inFilename):
    """
    Load a word dictionary as a tsv file (tab separated), which is what the tensorboard needs.
    NOTE: this returns a reverse dictionary i.e. wordid to word string lookup
    NOTE2: this would almost load a forward word dictionary as it just reads in the key,value pairs from the csv lines
    EXCEPT for the fact that it converts the dictionary value to an integer
    :param inFilename: filename containing the data
    :return:
    """
    reverse_dictionary = dict()
    with open(inFilename, 'rt') as f:
        reader = csv.reader(f, delimiter='\t', quotechar='"')
        next(reader) #skip header row which isn't detected as it's TSV format
        for row in reader:
            word = row[0]
            wordid = int(row[1])
            reverse_dictionary[wordid]=word
            #print(word,wordid)
    return reverse_dictionary

######################################################################################################

def wordsToWordIds(words,dictionary):
    """
    Given a list of words and a word->id dictionary, convert all the words to their id representation
    :param words: list of alpha words
    :param dictionary: lookup from word -> id
    :return: words converted to integer ids
    """
    return [dictionary[w] for w in words if w in dictionary] #how efficient is this pattern?

######################################################################################################

class DataProvider(object):
    """
    Provide raw data and targets for training. Data comes from a word file, which needs processing into a dictionary
    of words and then into a batch of word ids. All functions here are designed to work with raw input text data
    to provide a batch of training data and targets.
    """
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = self.makeBatch(data, batch_size, num_steps)

    ######################################################################################################

    def makeBatch(self, raw_data, batch_size, num_steps):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len],[batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = data[:, i * num_steps:(i + 1) * num_steps]
        x.set_shape([batch_size, num_steps])
        y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
        y.set_shape([batch_size, num_steps])
        return x, y

    ######################################################################################################


######################################################################################################

class Seq2Seq:
    def __init__(self, input, is_training, hidden_size, vocab_size, num_layers, dropout=0.5, init_scale=0.05):
        """
        Sequence to sequence model using word vectors, LSTM cells and dropout
        :param input: DataProvider class with all the information about the input data in it. The data has already
        been formatted as a tensor of int word ids at this point, so we need that and the number of batches and time
        step size used to format it. Provides a makeBatch function.
        :param is_training: True if we're in training mode
        :param hidden_size: cell size for the LSTM units
        :param vocab_size: number of words in the vocabulary dictionary (embedding)
        :param num_layers: number of deep stacked LSTM layers
        :param dropout: dropout rate for training to prevent overfitting see: tf.contrib.rnn.DropoutWrapper
        :param init_scale: word embedding vectors are initially created as uniform random withing +- this range
        """
        self.is_training = is_training #follow tf example of using same model code for training and testing
        self.input_obj = input #DataProvider
        self.batch_size = input.batch_size
        self.num_steps = input.num_steps #time steps
        self.hidden_size = hidden_size #LSTM unit cell size
        self.num_layers = num_layers #deep stacked LSTM layers

        # create the word embeddings
        with tf.device("/cpu:0"):
            embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale),name="embedding")
            inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)

        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        # set up the state storage / extraction
        self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size], name="LSTMState")

        state_per_layer_list = tf.unstack(self.init_state, axis=0) #axis 0 is layer, so split tensor into list of state for each layer
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) #s and h states
             for idx in range(num_layers)]
        )

        # create an LSTM cell to be unrolled
        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
        # add a dropout wrapper if training
        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)

        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)
        # reshape to (batch_size * num_steps, hidden_size)
        output = tf.reshape(output, [-1, hidden_size])

        softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -init_scale, init_scale))
        softmax_b = tf.Variable(tf.random_uniform([vocab_size], -init_scale, init_scale))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.input_obj.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        self.cost = tf.reduce_sum(loss)

        # get the prediction accuracy
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if not is_training:
            return
        self.learning_rate = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            #global_step=tf.contrib.framework.get_or_create_global_step()) #deprecated
            global_step=tf.train.get_or_create_global_step())
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.new_lr = tf.placeholder(tf.float32, shape=[], name="neta")
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    ######################################################################################################

    def assign_lr(self, session, lr_value):
        """Learning rate function"""
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

    ######################################################################################################


    #previous test functions...

    def build_dataset(self, words):
        count = collections.Counter(words).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reverse_dictionary

    def test_trainLMTest(training_data):
        """
        Taken and modified a bit from here:
        simpler LSTM example: https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/RNN/rnn_words.py
        Build an LSTM deep network.
        see: https://www.tensorflow.org/tutorials/recurrent
        :param lmodel language model i.e. gensim
        :param training_data block of text to learn
        :return:
        """

        dictionary, reverse_dictionary = self.build_dataset(training_data)
        vocab_size = len(dictionary)
        print("dict: ")
        for key in dictionary.keys():
            print(key)

        # Target log path
        logs_path = 'rnn_words'
        writer = tf.summary.FileWriter(logs_path)

        # Parameters
        learning_rate = 0.001
        training_iters = 50000
        display_step = 1000

        # inputs
        # embedding_matrix is a tensor of shape [vocabulary_size, embedding size]
        # word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)
        #word_embeddings = tf.nn.embedding_lookup(lmodel.values, lmodel.keys)  # I think?????

        #Method from Mikolov: take 1 hot words vector as input and use a projection matrix to project onto a lower
        #dimensional space (30?). Then replace his RNN with LSTM. Output is one hot.

        #wvector_size = 100 #size of word vector embeddings

        #lstm_size=100 #matches word vector dimension
        #number_of_layers = 5 #lstm network depth

        #initial values for batch training - tune later when you have more data
        #batch_size = 2
        time_steps = 3

        num_units=512 #number of units in lstm


        # tf Graph input
        x = tf.placeholder("float", [None, time_steps, 1])
        y = tf.placeholder("float", [None, vocab_size])

        # RNN output node weights and biases
        weights = {
            'out': tf.Variable(tf.random_normal([num_units, vocab_size]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([vocab_size]))
        }

        def RNN(x, weights, biases):
            # reshape to [1, n_input]
            x = tf.reshape(x, [-1, time_steps])

            # Generate a n_input-element sequence of inputs
            # (eg. [had] [a] [general] -> [20] [6] [33])
            x = tf.split(x, time_steps, 1)

            # 2-layer LSTM, each layer has n_hidden units.
            # Average Accuracy= 95.20% at 50k iter
            rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_units), rnn.BasicLSTMCell(num_units)])

            # 1-layer LSTM with n_hidden units but with lower accuracy.
            # Average Accuracy= 90.60% 50k iter
            # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
            # rnn_cell = rnn.BasicLSTMCell(n_hidden)

            # generate prediction
            outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

            # there are n_input outputs but
            # we only want the last output
            return tf.matmul(outputs[-1], weights['out']) + biases['out']

        pred = RNN(x, weights, biases)

        # Loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

        # Model evaluation
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        start_time = time.time()
        with tf.Session() as session:
            session.run(init)
            step = 0
            offset = random.randint(0, time_steps + 1)
            end_offset = time_steps + 1
            acc_total = 0
            loss_total = 0

            writer.add_graph(session.graph)

            while step < training_iters:
                # Generate a minibatch. Add some randomness on selection process.
                if offset > (len(training_data) - end_offset):
                    offset = random.randint(0, time_steps + 1)

                symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset + time_steps)]
                symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, time_steps, 1])

                symbols_out_onehot = np.zeros([vocab_size], dtype=float)
                symbols_out_onehot[dictionary[str(training_data[offset + time_steps])]] = 1.0
                symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

                _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                        feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
                loss_total += loss
                acc_total += acc
                if (step + 1) % display_step == 0:
                    print("Iter= " + str(step + 1) + ", Average Loss= " + \
                          "{:.6f}".format(loss_total / display_step) + ", Average Accuracy= " + \
                          "{:.2f}%".format(100 * acc_total / display_step))
                    acc_total = 0
                    loss_total = 0
                    symbols_in = [training_data[i] for i in range(offset, offset + time_steps)]
                    symbols_out = training_data[offset + time_steps]
                    symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                    print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
                step += 1
                offset += (time_steps + 1)
            print("Optimization Finished!")
            print("Elapsed time: ", time.time() - start_time)
            print("Run on command line.")
            print("\ttensorboard --logdir=%s" % (logs_path))
            print("Point your web browser to: http://localhost:6006/")
            while True:
                prompt = "%s words: " % time_steps
                sentence = input(prompt)
                sentence = sentence.strip()
                words = sentence.split(' ')
                if len(words) != time_steps:
                    continue
                try:
                    symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
                    for i in range(32):
                        keys = np.reshape(np.array(symbols_in_keys), [-1, time_steps, 1])
                        onehot_pred = session.run(pred, feed_dict={x: keys})
                        onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                        sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index])
                        symbols_in_keys = symbols_in_keys[1:]
                        symbols_in_keys.append(onehot_pred_index)
                    print(sentence)
                except:
                    print("Word not in dictionary")

###

def test_trainLMEmbedding(lmodel,training_data):
    """
    More advanced version of LSTM word prediction model using word embedding vectors.
    Accuracy comes out at about 80%, but this isn't comparable to the previous symbol version as it's mean square error
    on word vectors. Word accuracy is around 99%.
    :param lmodel:
    :param training_data:
    :return:
    """
    #Make the language model into an embedding lookup suitable for tf
    # embedding_matrix is a tensor of shape [vocabulary_size, embedding size]
    vocab = list(lmodel.wv.vocab)  # list of words in the vocabulary so we can get them in the right order
    embed = lmodel[vocab]
    embedding_matrix = tf.Variable(embed)
    word_ids = [i for i in range(0,len(vocab))] #[0,1,2,3...num_words-1]
    #dictionary = dict(vocab,word_ids) #word string to word id lookup
    dictionary = dict()
    for word in vocab:
        dictionary[word]=len(dictionary)
    reverse_dictionary = dict([[v,k] for k,v in dictionary.items()] ) #word id to word string lookup

    # Placeholders for inputs
    #embedded_word_ids = tf.nn.embedding_lookup(embedding_matrix, word_ids) #shape [vocab_size,word_vec_len], allows word id lookup
    #vocab_size = len(vocab) #derived value, how many words in vocabulary
    vocab_size, word_vec_len = embedding_matrix.shape  # get number of words in vocab and the word vector size
    vocab_size=int(vocab_size)
    word_vec_len=int(word_vec_len)

    # Target log path
    logs_path = 'rnn_words'
    writer = tf.summary.FileWriter(logs_path)

    # Parameters
    learning_rate = 0.001
    training_iters = 50000
    display_step = 1000

    #other params defining network size and config
    time_steps = 3
    num_units = 512  # number of units in lstm
    num_layers = 2 # number of layers in deep lstm

    # helpers for stacked lstm cells
    #def lstm_cell():
    #    return tf.contrib.rnn.BasicLSTMCell(num_units)

    #stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    #    [lstm_cell() for _ in range(num_layers)])

    # tf Graph input
    x = tf.placeholder("float", [None, time_steps * word_vec_len, 1]) #CHANGE! this is altered from (time_steps x int) to (time_steps x word_vec_len)
    y = tf.placeholder("float", [None, word_vec_len]) #CHANGE! this is altered from 1 hot to a 100 vector embedding

    # RNN output node weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([num_units, word_vec_len])) #CHANGE! was vocab_size for one hot, now word_vec_len
    }
    biases = {
        'out': tf.Variable(tf.random_normal([word_vec_len])) #CHANGE! as above
    }

    def RNN(x, weights, biases):
        # reshape to [1, n_input]
        x = tf.reshape(x, [-1, time_steps * word_vec_len])

        # Generate a n_input-element sequence of inputs
        # (eg. [had] [a] [general] -> [20] [6] [33])
        x = tf.split(x, time_steps, 1) #split x into (time_steps) sub tensors of word_vec_len elements each

        # 2-layer LSTM, each layer has n_hidden units.
        # Average Accuracy= 95.20% at 50k iter
        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_units), rnn.BasicLSTMCell(num_units)])

        # 1-layer LSTM with n_hidden units but with lower accuracy.
        # Average Accuracy= 90.60% 50k iter
        # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
        # rnn_cell = rnn.BasicLSTMCell(n_hidden)

        # generate prediction
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        # there are n_input outputs but
        # we only want the last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    pred = RNN(x, weights, biases)

    # Loss and optimizer - TODO: this needs to change
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) #this needs to change from logits
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    ##tf example
    # Compute the NCE loss, using a sample of the negative labels each time.
    #loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,labels=train_labels,inputs=embed,num_sampled=num_sampled,num_classes=vocabulary_size))
    # We use the SGD optimizer.
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
    #mine
    cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y,predictions=pred))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    start_time = time.time()
    with tf.Session() as session:
        session.run(init)
        step = 0
        offset = random.randint(0, time_steps + 1)
        end_offset = time_steps + 1
        acc_total = 0
        loss_total = 0

        writer.add_graph(session.graph)

        while step < training_iters:
            # Generate a minibatch. Add some randomness on selection process.
            if offset > (len(training_data) - end_offset):
                offset = random.randint(0, time_steps + 1)

            #symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset + time_steps)]
            #symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, time_steps, 1])
            input_words = [training_data[i] for i in range(offset, offset + time_steps)] # string version of input
            input_ids = []
            #build the input_vectors as a flat array of word vectors [w0,w1,w2...wn] where w0=100 element embed vector
            #input_vectors = np.arange(300)
            input_vectors = np.zeros(time_steps * word_vec_len)
            pos=0
            for word in input_words:
                if word in dictionary:
                    id = dictionary[word]
                    input_ids.append(id)
                    #input_vectors.append(embedded_word_ids[id])
                    input_vectors[pos:pos+word_vec_len]=embed[id]
                else:
                    input_ids.append(-1)  # using -1 as unknown word for now
                    #input_vectors.append(np.zeros([word_vec_len], dtype=float))
                    input_vectors[pos:pos+word_vec_len]=np.zeros([word_vec_len], dtype=float)
                pos=pos+word_vec_len
            #print(input_vectors)
            input_vectors = np.reshape(input_vectors,[-1,time_steps * word_vec_len,1])

            #now make the target value
            #symbols_out_onehot = np.zeros([vocab_size], dtype=float)
            #symbols_out_onehot[dictionary[str(training_data[offset + time_steps])]] = 1.0
            #symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])
            output_word = training_data[offset+time_steps]
            output_id = -1
            output_vector = np.zeros([word_vec_len], dtype=float)
            if output_word in dictionary:
                output_id = dictionary[output_word]
                #output_vector = embedded_word_ids[output_id]
                output_vector = embed[output_id]
            output_vector = np.reshape(output_vector, [1,word_vec_len])

            #this produces the right size output
            #output_vector = np.zeros(100)
            #output_vector = np.reshape(output_vector, [1,100])

            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                    feed_dict={x: input_vectors, y: output_vector})
            loss_total += loss
            acc_total += acc
            if (step + 1) % display_step == 0:
                print("Iter= " + str(step + 1) + ", Average Loss= " + \
                      "{:.6f}".format(loss_total / display_step) + ", Average Accuracy= " + \
                      "{:.2f}%".format(100 * acc_total / display_step))
                acc_total = 0
                loss_total = 0
                #TODO: fix this for the vector data
                #symbols_in = [training_data[i] for i in range(offset, offset + time_steps)]
                #symbols_out = training_data[offset + time_steps]
                #symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                #print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
                ###
                #print("%s - [%s] vs [%s]" % (input_ids, output_id, -1))
                output_word_pred = lmodel.most_similar( onehot_pred , [], 1)
                print("%s - [%s] vs [%s]" % (input_words, output_word, output_word_pred))
            step += 1
            offset += (time_steps + 1)
        print("Training finished!")
        print("Elapsed time: ", time.time() - start_time)



        #end

        #from the tensor flow example
        # words_in_dataset = tf.placeholder(tf.float32, [time_steps, batch_size, num_features]) #this is the training input to the net
        # lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size) # param is num_cells
        # # Initial state of the LSTM memory.
        # hidden_state = tf.zeros([batch_size, lstm.state_size])
        # current_state = tf.zeros([batch_size, lstm.state_size])
        # state = hidden_state, current_state
        # probabilities = []
        # loss = 0.0
        # for current_batch_of_words in words_in_dataset:
        #     # The value of state is updated after processing each batch of words.
        #     output, state = lstm(current_batch_of_words, state)
        #
        #
        #
        #     # The LSTM output can be used to make next word predictions
        #     logits = tf.matmul(output, softmax_w) + softmax_b
        #     probabilities.append(tf.nn.softmax(logits))
        #     loss += loss_function(probabilities, target_words)


######################################################################################################
#    More advanced model using word vector embeddings, stacked LSTM cells and softmax output.
#    This is based largely on the tensorflow embedding and seq2seq example, plus other examples online
# TODO: this really needs to go into the class
def trainLMAdvanced(idwords, vocabulary_size, num_layers, num_epochs, batch_size, model_save_name,
              learning_rate=1.0, max_lr_epoch=10, lr_decay=0.93, print_iter=50):
    #intermediate model outputs
    logs_path = "rnn_words"
    #TODO: and tensorboard
    writer = tf.summary.FileWriter(logs_path)

    # setup data and models
    training_input = DataProvider(batch_size=batch_size, num_steps=35, data=idwords)
    m = Seq2Seq(training_input, is_training=True, hidden_size=650, vocab_size=vocabulary_size,num_layers=num_layers)
    init_op = tf.global_variables_initializer()
    orig_decay = lr_decay
    with tf.Session() as session:
        # start threads
        session.run([init_op])
        writer.add_graph(session.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()
        for epoch in range(num_epochs):
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
            m.assign_lr(session, learning_rate * new_lr_decay)
            # m.assign_lr(sess, learning_rate)
            # print(m.learning_rate.eval(), new_lr_decay)
            current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))
            curr_time = dt.datetime.now()
            for step in range(training_input.epoch_size):
                # cost, _ = sess.run([m.cost, m.optimizer])
                if step % print_iter != 0:
                    cost, _, current_state = session.run([m.cost, m.train_op, m.state],
                                                          feed_dict={m.init_state: current_state})
                else:
                    seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
                    curr_time = dt.datetime.now()
                    cost, _, current_state, acc = session.run([m.cost, m.train_op, m.state, m.accuracy],
                                                               feed_dict={m.init_state: current_state})
                    print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}, Seconds per step: {:.3f}".format(
                            epoch,
                            step, cost, acc, seconds))

            # save a model checkpoint
            saver.save(session, logs_path + '\\' + model_save_name, global_step=epoch)
        # do a final save
        saver.save(session, logs_path + '\\' + model_save_name + '-final')
        # close threads
        coord.request_stop()
        coord.join(threads)

######################################################################################################

# load a previously trained model and test it
def testLMAdvanced(model_path, idwords, dictionary, reversed_dictionary):
    """
    Inject inputText into the model and see what the output is.
    TODO: we don't strictly need idwords, except for the fact that the training input is in the Seq2Seq constructor.
    It would be good to remove this, but it's tricky.
    :param model_path: Location of where a train version of the model is stored.
    :param idwords: tokensied version of the input corpus (is this needed?)
    :param dictionary: word -> id lookup so we can convert inputText into symbols
    :param reversed_dictionary: id -> word lookup so we can convert the output back into real words
    :param inputText: plain text input to the model which we are testing
    :return:
    """
    logs_path="rnn_words"
    #test_input = Input(batch_size=20, num_steps=35, data=test_data)
    #All this MUST match the parameters that it was trained with, otherwise bad things will happen
    num_words = len(reversed_dictionary.keys())
    training_input = DataProvider(batch_size=20, num_steps=35, data=idwords)
    m = Seq2Seq(training_input, is_training=False, hidden_size=650, vocab_size=num_words, num_layers=2)
    saver = tf.train.Saver()
    with tf.Session() as session:
        # start threads queue - it won't work if you don't do this
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        #load weights from the previous training session
        saver.restore(session, model_path)
        state = np.zeros((m.num_layers,2,m.batch_size,m.hidden_size))
        true_vals, pred, state, acc = session.run(
            [m.input_obj.targets, m.predict, m.state, m.accuracy],
            feed_dict={m.init_state: state}
        )
        #print("finished onehot=",onehot)
        #print("finished pred=", pred)
        print("shape(true_vals)=",true_vals.shape) #20,35
        print("shape(pred)=", pred.shape) #700
        pred_string = [reversed_dictionary[x] for x in pred[:m.num_steps]]
        true_vals_string = [reversed_dictionary[x] for x in true_vals[0]]
        print("True values (1st line) vs predicted values (2nd line):")
        print(" ".join(true_vals_string))
        print(" ".join(pred_string))
        coord.request_stop()
        coord.join(threads)

    #with tf.Session() as sess:
    #    # start threads
    #    coord = tf.train.Coordinator()
    #    threads = tf.train.start_queue_runners(coord=coord)
    #    current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))
    #    # restore the trained model
    #    saver.restore(sess, model_path)
    #    # get an average accuracy over num_acc_batches
    #    num_acc_batches = 8 #30
    #    check_batch_idx = 5 #25
    #    acc_check_thresh = 5
    #    accuracy = 0
    #    for batch in range(num_acc_batches):
    #        if batch == check_batch_idx:
    #            true_vals, pred, current_state, acc = sess.run([m.input_obj.targets, m.predict, m.state, m.accuracy],
    #                                                           feed_dict={m.init_state: current_state})
    #            pred_string = [reversed_dictionary[x] for x in pred[:m.num_steps]]
    #            true_vals_string = [reversed_dictionary[x] for x in true_vals[0]]
    #            print("True values (1st line) vs predicted values (2nd line):")
    #            print(" ".join(true_vals_string))
    #            print(" ".join(pred_string))
    #        else:
    #            acc, current_state = sess.run([m.accuracy, m.state], feed_dict={m.init_state: current_state})
    #        if batch >= acc_check_thresh:
    #            accuracy += acc
    #    print("Average accuracy: {:.3f}".format(accuracy / (num_acc_batches - acc_check_thresh)))
    #    # close threads
    #    coord.request_stop()
    #    coord.join(threads)





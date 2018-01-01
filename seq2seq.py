
import random
import time
import collections

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


#this is good on contextual chatbots with tf
#https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077

#simpler LSTM example: https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/RNN/rnn_words.py

class Seq2Seq:
    def __init__(self):
        self.text1=0

    ######################################################################################################

    def build_dataset(self, words):
        count = collections.Counter(words).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reverse_dictionary

    def trainLMTest(self, training_data):
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

    def trainLMEmbedding(self,lmodel,training_data):
        """
        More advanced version of LSTM word prediction model using word embedding vectors
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

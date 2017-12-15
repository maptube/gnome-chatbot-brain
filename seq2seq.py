
import tensorflow as tf


class Seq2Seq:
    def __init__(self):
        self.text1=0

    ######################################################################################################

    def trainLM(self,lmodel):
        """
        Build an LSTM deep network.
        see: https://www.tensorflow.org/tutorials/recurrent
        :param lmodel language model i.e. gensim
        :return:
        """

        lstm_size=100 #matches word vector dimension
        number_of_layers = 5 #lstm network depth

        #helpers for stacked lstm cells
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(lstm_size)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [lstm_cell() for _ in range(number_of_layers)])

        #inputs
        # embedding_matrix is a tensor of shape [vocabulary_size, embedding size]
        word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)


        words_in_dataset = tf.placeholder(tf.float32, [time_steps, batch_size, num_features])
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        # Initial state of the LSTM memory.
        hidden_state = tf.zeros([batch_size, lstm.state_size])
        current_state = tf.zeros([batch_size, lstm.state_size])
        state = hidden_state, current_state
        probabilities = []
        loss = 0.0
        for current_batch_of_words in words_in_dataset:
            # The value of state is updated after processing each batch of words.
            output, state = lstm(current_batch_of_words, state)

            # The LSTM output can be used to make next word predictions
            logits = tf.matmul(output, softmax_w) + softmax_b
            probabilities.append(tf.nn.softmax(logits))
            loss += loss_function(probabilities, target_words)

import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size

        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.lstm_layer = tf.keras.layers.LSTM(self.lstm_size, return_sequences=True, return_state=True)

    def call(self, sequence, states):
        emb = self.embedding_layer(sequence)
        output, state_h, state_c = self.lstm_layer(emb, initial_state=states)
        return output, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size

        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.lstm_layer = tf.keras.layers.LSTM(self.lstm_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, state):
        embed = self.embedding_layer(sequence)
        lstm_out, state_h, state_c = self.lstm_layer(embed, state)
        logits = self.dense(lstm_out)
        return logits, state_h, state_c

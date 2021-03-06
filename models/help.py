import tensorflow as tf
from models.attentions import LuongAttention, BahdanauAttention


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


class DecoderLuong(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size

        self.attention = LuongAttention(lstm_size)

        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm_layer = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True)
        self.dense_1 = tf.keras.layers.Dense(lstm_size, activation='tahn')
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, state, encoder_output):
        embed = self.embedding_layer(sequence)
        lstm_out, state_h, state_c = self.lstm_layer(embed, state)
        context, alignment = self.attention(lstm_out, encoder_output)
        lstm_out = tf.concat([tf.squeeze(context, 1), tf.squeeze(lstm_out, 1)], 1)

        lstm_out = self.dense_1(lstm_out)

        # Finally, it is converted back to vocabulary space: (batch_size, vocab_size)
        logits = self.dense(lstm_out)

        return logits, state_h, state_c, alignment


class BahdanauDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size

        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm_layer = tf.keras.layers.LSTM(lstm_size, return_state=True, return_sequences=True)
        self.attention = BahdanauAttention(lstm_size)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, seq, state, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(state, enc_output)

        seq = self.embedding_layer(seq)

        seq = tf.concat([tf.expand_dims(context_vector, 1), seq], axis=-1)

        seq, _, _ = self.lstm_layer(seq)
        output = tf.reshape(seq, (-1, seq.shape[2]))

        x = self.dense(output)

        return x, context_vector, attention_weights

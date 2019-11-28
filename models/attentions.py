import tensorflow as tf


class LuongAttention(tf.keras.Model):
    def __init__(self, size):
        super(LuongAttention, self).__init__()
        self.att = tf.keras.layers.Dense(size)

    def call(self, encoder_output, decoder_output):
        score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True)
        alignment = tf.nn.softmax(score, axis=2)
        context = tf.matmul(alignment, encoder_output)
        return context, alignment


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

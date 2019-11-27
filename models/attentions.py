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

import tensorflow as tf
from models.help import BahdanauDecoder, Encoder

class Seq2SeqBahdanau(object):

    model_name = 'Seq2Seq Bahdanau attention'

    def __init__(self, config, model_props):
        print('Creating %s with general config : %s and model config : %s' % (self.model_name, config, model_props))
        self.max_inp_seq_len = config['max_inp_seq_len']
        self.max_trg_seq_len = config['max_trg_seq_len']
        self.inp_vocab_size = config['inp_vocab_size']
        self.trg_vocab_size = config['trg_vocab_size']

        self.hidden_units = model_props['hidden_units']
        self.embedding_size = model_props['embedding_size']
        self.trainable_embeddings = model_props['trainable_embedding']
        self.save_model = model_props['save_model']
        self.save_model_plot = model_props['save_model_plot']
        self.epochs = model_props['epochs']

        self.encoder = Encoder(self.inp_vocab_size, self.embedding_size, self.hidden_units)
        self.decoder = BahdanauDecoder(self.inp_vocab_size, self.embedding_size, self.hidden_units)
        self.optimizer = tf.keras.optimizers.Adam()

    @staticmethod
    def loss_function(logits, targets):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        loss_ = loss_object(targets, logits)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def train_step(self, target_seq_in, target_seq_out, en_initial_states, batch_size, trg_tokenizer):
        loss = 0
        with tf.GradientTape() as tape:
            enc, enc_state = self.encoder(target_seq_in, en_initial_states)
            dec_hidden = enc_state

            dec_input = tf.expand_dims([trg_tokenizer.word_index['bos']] * batch_size, 1)
            for t in range(1, target_seq_out.shape[1]):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc)

                loss += Seq2SeqBahdanau.loss_function(target_seq_out[:, t], predictions)

                dec_input = tf.expand_dims(target_seq_out[:, t], 1)
            batch_loss = (loss / int(target_seq_out.shape[1]))
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return batch_loss

    def train(self, data_set, batch_size):
        for epochs in range(self.epochs):
            enc_hidden = self.encoder.init_states()
            total_loss = 0
            for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(data_set.take(-1)):
                if source_seq.shape[0] != batch_size:
                    print('Not enough samples for that epoch')
                    break
                loss = self.train_step(source_seq, target_seq_in, target_seq_out, enc_hidden)
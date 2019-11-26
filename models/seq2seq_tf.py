import tensorflow as tf
from models.help import Encoder, Decoder
from utils import load_tokenizers


class Seq2SeqBase(object):

    model_name = 'Seq2Seq TF'

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
        self.decoder = Decoder(self.trg_vocab_size, self.embedding_size, self.hidden_units)

    def loss_func(self, targets, logits):
        crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        mask = tf.cast(mask, dtype=tf.int64)
        loss = crossentropy(targets, logits, sample_weight=mask)
        return loss

    def train_step(self, source_seq, target_seq_in, target_seq_out, en_initial_states):
        with tf.GradientTape() as tape:
            en_outputs = self.encoder(source_seq, en_initial_states)
            en_states = en_outputs[1:]
            de_states = en_states

            de_outputs = self.decoder(target_seq_in, de_states)
            logits = de_outputs[0]
            loss = self.loss_func(target_seq_out, logits)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)

        optimizer = tf.keras.optimizers.RMSprop()

        optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def train(self, data_set, batch_size):
        print('Start training %s on %d' % (self.model_name, self.epochs))
        for e in range(self.epochs):
            inp_initial_states = self.encoder.init_states(batch_size)
            for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(data_set.take(-1)):
                if source_seq.shape[0] != batch_size:
                    print('Not enough samples for that epoch')
                    break
                loss = self.train_step(source_seq, target_seq_in, target_seq_out, inp_initial_states)
                print('Epoch {:02}   Step {:02}   Loss {:.4f}'.format(e, batch, loss))
            print('Epoch {} Loss {:.4f}'.format(e + 1, loss.numpy()))

    def predict(self, text):
        inp_tokenizer, trg_tokenizer = load_tokenizers()
        print(text)
        test_source_seq = inp_tokenizer.texts_to_sequences([text])
        print(test_source_seq)

        en_initial_states = self.encoder.init_states(1)
        en_outputs = self.encoder(tf.constant(test_source_seq), en_initial_states)

        de_input = tf.constant([[trg_tokenizer.word_index['<bos>']]])
        de_state_h, de_state_c = en_outputs[1:]
        out_words = []

        while True:
            de_output, de_state_h, de_state_c = self.decoder(
                de_input, (de_state_h, de_state_c))
            de_input = tf.argmax(de_output, -1)
            out_words.append(trg_tokenizer.index_word[de_input.numpy()[0][0]])

            if out_words[-1] == '<eos>' or len(out_words) >= 20:
                break

        print(' '.join(out_words))

    def test(self):
        source_input = tf.constant([[1, 2, 3, 4, 5, 0, 0, 0]])
        initial_state = self.encoder.init_states(1)
        encoder_output, en_state_h, en_state_c = self.encoder(source_input, initial_state)

        target_input = tf.constant([[1, 2, 3, 4, 5, 0, 0]])
        decoder_output, de_state_h, de_state_c = self.decoder(target_input, (en_state_h, en_state_c))

        print('Source sequences', source_input.shape)
        print('Encoder outputs', encoder_output.shape)
        print('Encoder state_h', en_state_h.shape)
        print('Encoder state_c', en_state_c.shape)

        print('\nDestination vocab size', self.trg_vocab_size)
        print('Destination sequences', target_input.shape)
        print('Decoder outputs', decoder_output.shape)
        print('Decoder state_h', de_state_h.shape)
        print('Decoder state_c', de_state_c.shape)

import numpy as np
import os
from keras.utils import to_categorical
from keras.layers import Input, LSTM, Embedding, Dense, CuDNNLSTM
from keras.models import Model, load_model
from keras.utils import plot_model
import constants
from utils import load_tokenizers


class Seq2Seq:
    model_name = 'seq2seq'

    def __init__(self, config, model_props):
        print('Creating %s with general config : %s and model config : %s' % (self.model_name, config, model_props))

        self.model = None
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

    def build_model(self):
        encoder_inputs = Input(shape=(self.max_inp_seq_len,))
        # English words embedding
        eng_encoder = Embedding(self.inp_vocab_size, self.embedding_size, input_length=self.max_inp_seq_len,
                                trainable=self.trainable_embeddings)(encoder_inputs)
        encoder_lstm_layer = LSTM(self.hidden_units, return_state=True, name='encoder_lstm_1')
        encoder_outputs, state_h, state_c = encoder_lstm_layer(eng_encoder)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(self.max_trg_seq_len,))

        # french word embeddings
        fra_emb_layer = Embedding(self.trg_vocab_size, self.embedding_size, trainable=self.trainable_embeddings)
        fra_emb = fra_emb_layer(decoder_inputs)

        # decoder lstm
        decoder_lstm_layer = LSTM(self.hidden_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm_layer(fra_emb, initial_state=encoder_states)

        decoder_dense = Dense(self.trg_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # rmsprop is preferred for nlp tasks
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

        # if self.save_model_plot:
        #     file = os.path.join(constants.PLOT_MODELS_FOLDER, self.model_name + '.png')
        #     plot_model(self.model, file, show_shapes=True, show_layer_names=True)

    def create_generator(self, x, y, batch_size=constants.DEFAULT_BATCH_SIZE):
        num_batches = len(x) // batch_size
        while True:
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                enc_data = x[start:end]
                dec_inp_data = y[start:end]
                decoder_target_data = np.zeros(
                    (batch_size, self.max_trg_seq_len, self.trg_vocab_size), dtype='float32')
                for seq_idx, seq in enumerate(dec_inp_data):
                    for word_idx, word in enumerate(seq):
                        if word_idx > 0:
                            oh_enc_word = to_categorical(word, num_classes=self.trg_vocab_size)
                            decoder_target_data[seq_idx, word_idx - 1] = oh_enc_word
                yield [enc_data, dec_inp_data], decoder_target_data

    def fit(self, x, y, x_val, y_val, batch_size):
        print('Going to train the %s on %d train samples and %d validation samples' % (self.model_name,
                                                                                       len(x), len(y)))
        train_generator = self.create_generator(x, y, batch_size)
        val_generator = self.create_generator(x_val, y_val, batch_size)

        num_training_batches = len(x) // batch_size
        num_val_batches = len(x_val) // batch_size

        self.model.fit_generator(generator=train_generator, steps_per_epoch=num_training_batches,
                                 epochs=self.epochs, validation_data=val_generator, validation_steps=num_val_batches)
        if self.save_model:
            print('Saving %s model' % self.model_name)
            file = self.get_file_name()
            self.model.save(file)
            print('Model saved.')

    def get_file_name(self):
        return os.path.join(constants.SAVED_MODELS_FOLDER, self.model_name + '.h5')

    def load_model(self):
        print('Loading model %s...' % self.model_name)
        self.model = load_model(self.get_file_name())
        print('Model loaded.')

    def get_encoder_and_decoder_models(self):
        encoder_inputs = self.model.get_layer('input_1').output
        print(encoder_inputs)
        print(self.model.get_layer('embedding'))
        encoder_emb = self.model.get_layer('embedding')(encoder_inputs)

        encoder = self.model.get_layer('encoder_lstm_1')
        encoder_outputs, state_h, state_c = encoder(encoder_emb)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        encoder_model = Model(encoder_inputs, encoder_states)

        # Decoder model
        decoder_state_input_h = Input(shape=(self.hidden_units,))
        decoder_state_input_c = Input(shape=(self.hidden_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_inputs_single = Input(shape=(1,))
        final_dex2 = self.model.get_layer('embedding_1')(decoder_inputs_single)
        decoder_outputs2, state_h2, state_c2 = self.model.get_layer('cu_dnnlstm')(final_dex2,
                                                                                  initial_state=decoder_states_inputs)
        decoder_states2 = [state_h2, state_c2]
        decoder_outputs2 = self.model.get_layer('dense')(decoder_outputs2)
        """sampling model will take encoder states and decoder_input(seed initially) and output the
         predictions(target word index) We don't care about decoder_states2"""
        decoder_model = Model(
            [decoder_inputs_single] + decoder_states_inputs,
            [decoder_outputs2] + decoder_states2)

        return encoder_model, decoder_model

    def predict(self, input_seq):
        encoder_model, decoder_model = self.get_encoder_and_decoder_models()
        _, trg_tokenizer = load_tokenizers()
        states_value = encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1))
        print('<BOS> index : ', trg_tokenizer.word_index['bos'])
        target_seq[0, 0] = trg_tokenizer.word_index['bos']
        eos = trg_tokenizer.word_index['eos']
        print('<EOS> index : ', eos)
        output_sentence = []

        for _ in range(self.max_trg_seq_len):
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            idx = np.argmax(output_tokens[0, 0, :])

            if eos == idx:
                break

            word = ''

            if idx > 0:
                word = trg_tokenizer.index_word[idx]
                output_sentence.append(word)

            target_seq[0, 0] = idx
            states_value = [h, c]

        return ' '.join(output_sentence)

import pandas as pd
import os
import constants
import pickle
from sklearn.utils import shuffle
import errno
import re
import unicodedata
import tensorflow as tf


def load_data(file=constants.DATA_FILE):
    print('Loading data from %s...' % file)
    if not os.path.exists(file):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), file)
    df = pd.read_csv(file, sep='\t', header=None, names=['eng', 'fra'])
    df = shuffle(df)
    df = df[:constants.WORKING_SAMPLES]
    print('Data loaded')
    return df


def clear_duplicates(df):
    print('Cleaning duplicates...')
    df.drop_duplicates(subset="eng", keep="first", inplace=True)
    print('Duplicates cleaned.')
    return df


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s


def preprocess():
    df = load_data()
    df = clear_duplicates(df)

    raw_input, raw_output = df['eng'].values, df['fra'].values
    data_inp = [normalize_string(r) for r in raw_input]
    data_trg_inp = [constants.START_TOKEN + normalize_string(r) for r in raw_output]
    data_trg_out = [normalize_string(r) + constants.END_TOKEN for r in raw_output]

    inp_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    trg_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

    print('Fit tokenizers to texts...')
    inp_tokenizer.fit_on_texts(data_inp)
    trg_tokenizer.fit_on_texts(data_trg_inp + data_trg_out)
    print('Tokenizers fit.')

    print('Saving the tokenizers...')
    with open(constants.INP_TOKENIZER_PATH, 'wb') as f:
        pickle.dump(inp_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(constants.TRG_TOKENIZER_PATH, 'wb') as f:
        pickle.dump(trg_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Tokenizers saved.')

    print('Texts to sequences...')
    inp_tok = inp_tokenizer.texts_to_sequences(data_inp)
    out_tok_in = trg_tokenizer.texts_to_sequences(data_trg_inp)
    out_tok_out = trg_tokenizer.texts_to_sequences(data_trg_out)

    print('Padding sequences...')
    inp_processed = tf.keras.preprocessing.sequence.pad_sequences(inp_tok, padding='post')
    out_in_processed = tf.keras.preprocessing.sequence.pad_sequences(out_tok_in, padding='post')
    out_out_processed = tf.keras.preprocessing.sequence.pad_sequences(out_tok_out, padding='post')

    inp_seq_len = inp_processed.shape[1]
    trg_seq_len = out_in_processed.shape[1]
    inp_vocab_size = len(inp_tokenizer.word_index) + 1
    trg_vocab_size = len(trg_tokenizer.word_index) + 1

    config = {
        'max_inp_seq_len': inp_seq_len,
        'max_trg_seq_len': trg_seq_len,
        'inp_vocab_size': inp_vocab_size,
        'trg_vocab_size': trg_vocab_size,
    }
    print('Config : ', config)

    print('Saving config...')
    with open(constants.CONFIG_FILE_PATH, 'wb') as f:
        pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Config saved.')

    print('Saving processed data...')
    with open(constants.INPUT_PROCESSED_DATA_PATH, 'wb') as f:
        pickle.dump(inp_processed, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(constants.TARGET_INPUT_PROCESSED_DATA_PATH, 'wb') as f:
        pickle.dump(out_in_processed, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(constants.TARGET_PROCESSED_DATA_PATH, 'wb') as f:
        pickle.dump(out_out_processed, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Processed data saved.')
    # ds = tf.data.Dataset.from_tensor_slices((inp_processed, out_in_processed, out_out_processed))
    # ds = ds.shuffle(20).batch(5)
    #
    # print('Saving data set...')
    # with open(constants.DATASET_FILE_PATH, 'wb') as f:
    #     pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)
    # print('Data set saved.')


if __name__ == '__main__':
    preprocess()

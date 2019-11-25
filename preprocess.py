import pandas as pd
import os
import errno
import constants
from utils import clean_text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.utils import shuffle


def load_data(file=constants.DATA_FILE):
    if not os.path.exists(file):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), file)
    df = pd.read_csv(file, sep='\t', header=None, names=['eng', 'fra'])
    df = shuffle(df)
    df = df[:constants.WORKING_SAMPLES]
    return df


def clean_data_frame(df):
    """Remove duplicated english sequences"""

    df.drop_duplicates(subset="eng", keep="first", inplace=True)
    df['eng'] = df['eng'].apply(clean_text)
    df['fra'] = df['fra'].apply(lambda x: constants.START_TOKEN + clean_text(x) + constants.END_TOKEN)
    return df


def tokenize(df):
    eng_tokenizer = Tokenizer(num_words=constants.NUM_INP_WORDS, oov_token=constants.UNK_TOKEN)
    fra_tokenizer = Tokenizer(num_words=constants.NUM_TRG_WORDS, oov_token=constants.UNK_TOKEN)

    eng_tokenizer.fit_on_texts(df['eng'])
    fra_tokenizer.fit_on_texts(df['fra'])

    print('Saving the tokenizers...')
    with open(constants.INP_TOKENIZER_PATH, 'wb') as f:
        pickle.dump(eng_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(constants.TRG_TOKENIZER_PATH, 'wb') as f:
        pickle.dump(fra_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Tokenizers saved.')
    return eng_tokenizer, fra_tokenizer


def padding(df, eng_tokenizer, fra_tokenizer):
    eng_vocab_size = min(constants.NUM_INP_WORDS, len(eng_tokenizer.word_index) + 1)
    fra_vocab_size = min(constants.NUM_TRG_WORDS, len(fra_tokenizer.word_index) + 1)
    """Add 1 to word_index because indexes start from 1(0 is for padding)"""

    print('Eng vocab size : ', eng_vocab_size)
    print('Fra vocab size : ', fra_vocab_size)

    """Tokenizing"""
    print('Tokenizing...')
    eng_sent_tokens = eng_tokenizer.texts_to_sequences(df['eng'])
    fra_sent_tokens = fra_tokenizer.texts_to_sequences(df['fra'])
    print('Tokenizing done.')

    max_eng_seq_len = max([len(x) for x in eng_sent_tokens])
    max_fra_seq_len = max([len(x) for x in fra_sent_tokens])

    print('Max eng seq len : ', max_eng_seq_len)
    print('Max fra seq len : ', max_fra_seq_len)

    print('Padding...')
    eng_padded = pad_sequences(eng_sent_tokens, maxlen=max_eng_seq_len, padding='post')
    fra_padded = pad_sequences(fra_sent_tokens, maxlen=max_fra_seq_len, padding='post')
    print('Padding done.')

    print('Saving processed sequences...')
    with open(constants.INP_PADDED_FILE, 'wb') as f:
        pickle.dump(eng_padded, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(constants.TRG_PADDED_FILE, 'wb') as f:
        pickle.dump(fra_padded, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Preprocessed sequences saved.')

    config = {
        'max_inp_seq_len': max_eng_seq_len,
        'max_trg_seq_len': max_fra_seq_len,
        'inp_vocab_size': eng_vocab_size,
        'trg_vocab_size': fra_vocab_size,
    }
    print('Config : ', config)

    print('Saving config...')
    with open(constants.CONFIG_FILE_PATH, 'wb') as f:
        pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Config saved.')


def pre_process():
    df = load_data(constants.DATA_FILE)
    df = clean_data_frame(df)
    eng_tokenizer, fra_tokenizer = tokenize(df)
    padding(df, eng_tokenizer, fra_tokenizer)


if __name__ == "__main__":
    pre_process()

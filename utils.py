import re
import pickle
import constants


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", " ", text)
    text = text.strip()
    return text


def load_tokenizers():
    print('Loading tokenizers...')
    inp_tokenizer = pickle.load(open(constants.INP_TOKENIZER_PATH, 'rb'))
    trg_tokenizer = pickle.load(open(constants.TRG_TOKENIZER_PATH, 'rb'))
    print('Tokenizers loaded.')
    return inp_tokenizer, trg_tokenizer


def load_config():
    print('Loading config...')
    config = pickle.load(open(constants.CONFIG_FILE_PATH, mode='rb'))
    print('Config loaded.')
    return config


def load_preprocessed_data():
    print('Loading input and target...')
    inp = pickle.load(open(constants.INP_PADDED_FILE, 'rb'))
    trg = pickle.load(open(constants.TRG_PADDED_FILE, 'rb'))
    print('Input and target loaded.')
    return inp, trg

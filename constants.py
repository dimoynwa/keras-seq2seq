DATA_FILE = './data/fra-eng.txt'

CONFIG_FILE_PATH = './data/config.pickle'

WORKING_SAMPLES = 30000

START_TOKEN = '<BOS> '
END_TOKEN = ' <EOS>'
UNK_TOKEN = '<UNK>'

NUM_INP_WORDS = 12000
NUM_TRG_WORDS = 20000

INP_PADDED_FILE = './data/inp_padded.pickle'
TRG_PADDED_FILE = './data/trg_padded.pickle'

INP_TOKENIZER_PATH = './data/eng_tokenizer.pickle'
TRG_TOKENIZER_PATH = './data/trg_tokenizer.pickle'

PLOT_MODELS_FOLDER = './plot_models/'
SAVED_MODELS_FOLDER = './saved_models/'

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS_NUMBER = 20
DEFAULT_HIDDEN_UNITS = 256
DEFAULT_EMBEDDING_SIZE = 100

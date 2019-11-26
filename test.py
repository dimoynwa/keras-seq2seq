import tensorflow as tf
from utils import load_config, load_dataset
from models.seq2seq_tf import Seq2SeqBase
import pickle
import constants

if __name__ == '__main__':
    config = load_config()
    model_props = {
        'hidden_units': 128,
        'embedding_size': 200,
        'trainable_embedding': True,
        'save_model': True,
        'save_model_plot': True,
        'epochs': 20,
    }

    seq2seq = Seq2SeqBase(config, model_props)
    seq2seq.test()

    input_data = pickle.load(open(constants.INPUT_PROCESSED_DATA_PATH, 'rb'))
    target_inp_data = pickle.load(open(constants.TARGET_INPUT_PROCESSED_DATA_PATH, 'rb'))
    target_data = pickle.load(open(constants.TARGET_PROCESSED_DATA_PATH, 'rb'))

    batch_size = constants.DEFAULT_BATCH_SIZE

    data = tf.data.Dataset.from_tensor_slices((input_data, target_inp_data, target_data))
    data = data.shuffle(20).batch(batch_size)

    seq2seq.train(data, batch_size)

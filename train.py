import argparse
from models.Seq2Seq import Seq2Seq
import constants
from utils import load_config, load_preprocessed_data
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reading command line arguments')
    parser.add_argument('--hidden_units', type=int, default=constants.DEFAULT_HIDDEN_UNITS)
    parser.add_argument('--epochs', type=int, default=constants.DEFAULT_EPOCHS_NUMBER)
    parser.add_argument('--embedding_size', type=int, default=constants.DEFAULT_EMBEDDING_SIZE)
    parser.add_argument('--train_emb', type=bool, default=False, help='Should the model train its embedding layer.')
    parser.add_argument('--save_model', type=bool, default=True, help='Do you want to save the model after the train')
    parser.add_argument('--save_plot_model', type=bool, default=True, help='Do you want to save the model plot'
                                                                           ' after the train')
    parser.add_argument('--batch_size', type=int, default=constants.DEFAULT_BATCH_SIZE)

    args = parser.parse_args()

    model_props = {
        'hidden_units': args.hidden_units,
        'embedding_size': args.embedding_size,
        'trainable_embedding': args.train_emb,
        'save_model': args.save_model,
        'save_model_plot': args.save_plot_model,
        'epochs': args.epochs,
    }

    inp, output = load_preprocessed_data()

    X_train, X_test, y_train, y_test = train_test_split(inp, output, test_size=0.2, random_state=42)

    config = load_config()
    seq2seq = Seq2Seq(config, model_props)
    seq2seq.build_model()

    seq2seq.fit(X_train, y_train, X_test, y_test, args.batch_size)

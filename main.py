#!/usr/bin/env python3
# coding: utf-8

'''
Luis A. Leiva, Moises Diaz, Miguel A. Ferrer, Réjean Plamondon.
Human or Machine? It Is Not What You Write, But How You Write It.
Proc. ICPR, 2020.

Train various DL models to classify human and synthetic handwriting movements.

External dependencies, to be installed e.g. via pip:
- numpy v1.16
- tensorflow v2.0
- sklearn v0.21

Author: Luis A. Leiva <luis.leiva@aalto.fi>
Last modified: 22.07.2020
'''

# Load std libs.
import sys
import os
import argparse
import math
import random
from time import time

# Configure CLI parser early.
parser = argparse.ArgumentParser(description='Train (or test) a CNN or RNN model \
  to classify human and synthetic handwriting movements',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Define CLI options.
parser.add_argument('--human_dir', help='path to human data directory')
parser.add_argument('--synth_dir', help='path to synth data directory')
parser.add_argument('--human_files', nargs='+', help='path to human files')
parser.add_argument('--synth_files', nargs='+', help='path to synth files')
parser.add_argument('--summary', action='store_true', help='display model architecture, memory usage, flops, etc.')
parser.add_argument('--sort_files', action='store_true', help='sort data, to promote test users not to be seen during model training')
parser.add_argument('--model_type', required=True, choices=['cnn', 'vgg', 'resnet', 'densenet', 'inception', 'xception', 'rnn', 'lstm', 'gru'], help='model type')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('--patience', default=10, type=int, help='number of consecutive epochs without improvement to stop training')
parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
parser.add_argument('--training_ratio', default=0.7, type=float, help='training partition size')
parser.add_argument('--validation_ratio', default=0.2, type=float, help='validation partition size, relative to training partition')
parser.add_argument('--activation', choices=['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential'], default='relu', help='activation function in hidden layers. NB: the output layer has ALWAYS sigmoid activation')
parser.add_argument('--eval_model', help='path to model file, for evaluation')
parser.add_argument('--workers', default=1, type=int, help='number of workers to parallelize training')
parser.add_argument('--out_dir', help='path to write TF logs and model')
parser.add_argument('--verbose', default=0, type=int, help='display more information to stdout')

# Define CNN-only options.
cnn_op = parser.add_argument_group('CNN options')
cnn_op.add_argument('--model_weights', help='model weights: either "imagenet" or an h5 filename')
cnn_op.add_argument('--image_size', default=160, help='image size, in px, for both width and height')
cnn_op.add_argument('--grayscale', action='store_true', help='use grayscale images')

# Define RNN-only options.
rnn_op = parser.add_argument_group('RNN options')
rnn_op.add_argument('--bidirectional', action='store_true', help='use bidirectional cells')
rnn_op.add_argument('--max_length', default=400, type=int, help='max number of timesteps')
rnn_op.add_argument('--chunk_length', default=0, type=int, help='split each sequence in shorter chunks')
rnn_op.add_argument('--sampling_rate', default=100, type=int, help='sampling rate, in Hz, if the dataset does not have a "time" column')
rnn_op.add_argument('--rnn_units', default=100, type=int, help='dimensionality of output embedding')

args = parser.parse_args()
if args.verbose:
    print(args, file=sys.stderr)

# Display only TF errors, if any.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load 3rd party libs.
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import eval_binary_classifier, get_model_memory_usage, get_model_flops

# Use current timestamp to keep individual runs of each experiment.
NOW = int(time())

# Boost GPU usage.
conf = tf.compat.v1.ConfigProto()
conf.gpu_options.allow_growth = True
tf.compat.v1.Session(config=conf)

# Makes results reproducible.
my_seed = 123456
random.seed(my_seed)
np.random.seed(my_seed)
tf.random.set_seed(my_seed)

# Silly checks.
if (args.human_dir and not os.path.isdir(args.human_dir)) or (args.synth_dir and not os.path.isdir(args.synth_dir)):
    print('Input data not found! Please revise your human and synth directories.')
    parser.print_help()
    exit()


def get_label(filepath):
    return os.path.basename(os.path.dirname(filepath))


def read_image(filepath, grayscale=False):
    image = tf.keras.preprocessing.image.load_img(filepath)
    image = np.asarray(image)
    image = tf.image.central_crop(image, 0.5)
    image = tf.image.resize(image, IMAGE_SIZE, method='nearest')
    if grayscale:
        image = tf.image.rgb_to_grayscale(image)
    return image


def read_sequence(filepath, is_synth):
    moves = []
    with open(filepath, 'rb') as f:
        lines = f.read().decode(errors='ignore').splitlines()
        lines.pop(0) # skip header

        try:
            points = read_points(lines)
        except:
            return None

        for i in range(1, len(points)):
            curr_pt = points[i]
            prev_pt = points[i - 1]
            # Unpack tuples.
            curr_x, curr_y, curr_t = curr_pt
            prev_x, prev_y, prev_t = prev_pt

            delta_x = curr_x - prev_x
            delta_y = curr_y - prev_y
            delta_t = curr_t - prev_t

            # Some datasets, like $N, have duplicated timestamps, so ignore them.
            # Also ignore negative timestamps (if any).
            if delta_t <= 0:
                continue

            # Use speed as the only discriminative feature, since:
            # (1) it's translation and rotation invariant;
            # (2) the SLM model uses velocity as motor control parameter,
            # so we can challenge the model better.
            speed = math.sqrt(delta_x**2 + delta_y**2) / delta_t
            moves.append([speed])
    return np.array(moves)


def read_points(lines):
    points = []
    # If no time information is available, assume constant sampling rate.
    # For 100 Hz, we have 1e3 * 1/100 = 10 ms between timesteps.
    rate = 0
    rinc = 1000/args.sampling_rate
    for i, line in enumerate(lines):
        cols = line.split()
        num_cols = len(cols)

        if num_cols < 3:
            raise ValueError('CSV lines must have at least 3 columns (found {})'.format(num_cols))

        if num_cols < 5:
            # No time column assumed.
            stroke_id, x, y, is_writing = cols[0:4]
            t = rate
        else:
            # Default columns assumed.
            stroke_id, x, y, t, is_writing = cols[0:5]

        rate += rinc
        # Cast values.
        x, y, t, is_writing = float(x), float(y), float(t), bool(float(is_writing))
        if is_writing:
            points.append([x, y, t])

    # TODO: Ensure that ALL our functions return numpy arrays.
    return points


def list_files(directory):
    files = []
    for r, d, f in os.walk(directory):
        for file in f:
            # Only accept PNG (for CNN model) and CSV (for RNN model).
            # NB: Assume that there is only PNG or CSV files in any directory.
            if file.endswith('.png') or file.endswith('.csv'):
                files.append(os.path.join(r, file))
    return files


def image_loader(files, batch_size=None, grayscale=False):
    if batch_size is not None:
        files = np.random.choice(files, size=batch_size)

    batch_input, batch_output = [], []
    for filepath in files:
        label = LABELS[get_label(filepath)]
        image = read_image(filepath, grayscale)

        batch_input.append(tf.keras.preprocessing.image.img_to_array(image))
        batch_output.append(label)

    X, y = np.array(batch_input), np.array(batch_output)
    return X, y


def sequence_loader(files, batch_size=None):
    if batch_size is not None:
        files = np.random.choice(files, size=batch_size)

    batch_input, batch_output = [], []
    for filepath in files:
        label = LABELS[get_label(filepath)]
        moves = read_sequence(filepath, bool(label))
        if moves is None:
            continue

        if args.chunk_length > 0:
            for seq in chunk_generator(moves, args.chunk_length):
                batch_input.append(seq)
                batch_output.append(label)
        else:
            batch_input.append(moves)
            batch_output.append(label)

    X, y = np.array(batch_input), np.array(batch_output)

    if args.max_length > 0:
        X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=args.max_length, dtype='float32')
        # Ensure we pass in the right data type. Is this is a TF2 issue?
        X = tf.cast(X, tf.float32).numpy()
        y = tf.cast(y, tf.int32).numpy()

    return X, y


def chunk_generator(points, n):
    for i in range(0, len(points), n):
        yield points[i:i + n]


def create_cnn_model(shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 3, padding='same', activation=args.activation, input_shape=shape),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation=args.activation),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2048, activation=args.activation),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_pretrained_cnn_model(base_model):
    x = base_model.output

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(4096, activation=args.activation)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    return model


def create_vgg_model(shape, weights):
    base_model = tf.keras.applications.VGG16(include_top=False, input_shape=shape, weights=weights)
    base_model.trainable = False
    return create_pretrained_cnn_model(base_model)


def create_resnet_model(shape, weights):
    base_model = tf.keras.applications.ResNet50V2(include_top=False, input_shape=shape, weights=weights)
    #base_model.trainable = False # Not beneficial
    return create_pretrained_cnn_model(base_model)


def create_densenet_model(shape, weights):
    base_model = tf.keras.applications.DenseNet201(include_top=False, input_shape=shape, weights=weights)
    #base_model.trainable = False # Not beneficial
    return create_pretrained_cnn_model(base_model)


def create_inception_model(shape, weights):
    base_model = tf.keras.applications.InceptionV3(include_top=False, input_shape=shape, weights=weights)
    #base_model.trainable = False # Not beneficial
    return create_pretrained_cnn_model(base_model)


def create_xception_model(shape, weights):
    base_model = tf.keras.applications.Xception(include_top=False, input_shape=shape, weights=weights)
    #base_model.trainable = False # Not beneficial
    return create_pretrained_cnn_model(base_model)


def create_rnn_model(shape, bidirectional=False):
    model = tf.keras.Sequential([ tf.keras.layers.Input(shape=shape) ])
    recurrent_block = tf.keras.layers.SimpleRNN(args.rnn_units, activation=args.activation)
    if bidirectional:
        recurrent_block = tf.keras.layers.Bidirectional(recurrent_block)
    model.add(recurrent_block)
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


def create_lstm_model(shape, bidirectional=False):
    model = tf.keras.Sequential([ tf.keras.layers.Input(shape=shape) ])
    recurrent_block = tf.keras.layers.LSTM(args.rnn_units, activation=args.activation)
    if bidirectional:
        recurrent_block = tf.keras.layers.Bidirectional(recurrent_block)
    model.add(recurrent_block)
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


def create_gru_model(shape, bidirectional=False):
    model = tf.keras.Sequential([ tf.keras.layers.Input(shape=shape) ])
    recurrent_block = tf.keras.layers.GRU(args.rnn_units, activation=args.activation)
    if bidirectional:
        recurrent_block = tf.keras.layers.Bidirectional(recurrent_block)
    model.add(recurrent_block)
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


# General configuration.
IMAGE_SIZE = (args.image_size, args.image_size)
OUTPUT_DIR = '/tmp/model-{}-{}'.format(args.model_type, NOW)

# Overwrite output dir, if set.
if args.out_dir and os.path.isdir(args.out_dir):
    OUTPUT_DIR = args.out_dir

# Load filenames.
human_files = list_files(args.human_dir) if args.human_dir else args.human_files
synth_files = list_files(args.synth_dir) if args.synth_dir else args.synth_files

# Map directory names to class names.
LABELS = {
  get_label(human_files[0]) : 0,
  get_label(synth_files[0]) : 1,
}
print('Classes:', LABELS, file=sys.stderr)

if args.sort_files:
    # Interleave human/synth data while ensuring file order,
    # so that in the later splits some users are not seen in training.
    human_files = sorted(human_files)
    synth_files = sorted(synth_files)
    train_files = np.array([f for files in zip(human_files, synth_files) for f in files])
else:
    # Collect data at random.
    train_files = np.array(human_files + synth_files)
    np.random.shuffle(train_files)

# Reserve some part of the training data for testing.
# Later we split the training data again for validation.
train_files, test_files = train_test_split(train_files, train_size=args.training_ratio,
  random_state=my_seed, shuffle=not(args.sort_files))

if args.eval_model:
    # Load existing model.
    # No need to compile since we'll be predicting test samples.
    model = tf.keras.models.load_model(args.eval_model, compile=False)

    if args.model_type == 'cnn':
        X_train, y_train = image_loader(train_files, grayscale=args.grayscale)
        X_test, y_test = image_loader(test_files, grayscale=args.grayscale)
    elif args.model_type in ['vgg', 'resnet', 'densenet', 'inception', 'xception']:
        X_train, y_train = image_loader(train_files)
        X_test, y_test = image_loader(test_files)
    elif args.model_type in ['rnn', 'lstm', 'gru']:
        X_train, y_train = sequence_loader(train_files)
        X_test, y_test = sequence_loader(test_files)

else:
    # Train model.
    # Notice that each model architecture requires different data loaders.
    if args.model_type == 'cnn':

        X_train, y_train = image_loader(train_files, grayscale=args.grayscale)
        X_test, y_test = image_loader(test_files, grayscale=args.grayscale)

        model = create_cnn_model(X_train[0].shape)

    elif args.model_type == 'vgg':

        X_train, y_train = image_loader(train_files)
        X_test, y_test = image_loader(test_files)

        model = create_vgg_model(X_train[0].shape, args.model_weights)

    elif args.model_type == 'resnet':

        X_train, y_train = image_loader(train_files)
        X_test, y_test = image_loader(test_files)

        model = create_resnet_model(X_train[0].shape, args.model_weights)

    elif args.model_type == 'densenet':

        X_train, y_train = image_loader(train_files)
        X_test, y_test = image_loader(test_files)

        model = create_densenet_model(X_train[0].shape, args.model_weights)

    elif args.model_type == 'inception':

        X_train, y_train = image_loader(train_files)
        X_test, y_test = image_loader(test_files)

        model = create_inception_model(X_train[0].shape, args.model_weights)

    elif args.model_type == 'xception':

        X_train, y_train = image_loader(train_files)
        X_test, y_test = image_loader(test_files)

        model = create_xception_model(X_train[0].shape, args.model_weights)

    elif args.model_type == 'rnn':

        X_train, y_train = sequence_loader(train_files)
        X_test, y_test = sequence_loader(test_files)

        model = create_rnn_model(X_train[0].shape, args.bidirectional)

    elif args.model_type == 'lstm':

        X_train, y_train = sequence_loader(train_files)
        X_test, y_test = sequence_loader(test_files)

        model = create_lstm_model(X_train[0].shape, args.bidirectional)

    elif args.model_type == 'gru':

        X_train, y_train = sequence_loader(train_files)
        X_test, y_test = sequence_loader(test_files)

        model = create_gru_model(X_train[0].shape, args.bidirectional)

    # All models should be compiled the same way, to make results comparable.
    # NB: The default LR seems too aggressive for this task, so recommended values are between 0.0005 and 0.0001.
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(args.lr), metrics=['accuracy'])

    # AS of TF2.0, the `validation_split` arg does NOT work with generators.
    # So if we can put all the dataset into memory, let's do it.
    # Otherwise we have to create a validation partition and use custom generators.
    model.fit(
        X_train,
        y_train,
        verbose = args.verbose,
        use_multiprocessing = True,
        workers = args.workers,
        epochs = args.epochs,
        batch_size = args.batch_size,
        validation_split = args.validation_ratio,
        callbacks = [
            tf.keras.callbacks.TensorBoard(OUTPUT_DIR, profile_batch=0),
            tf.keras.callbacks.EarlyStopping(patience=args.patience, verbose=args.verbose,
              restore_best_weights=True, monitor='val_accuracy', mode='max')
        ]
    )

    # Save model once training is done.
    model_file = os.path.join(OUTPUT_DIR, '{}.h5'.format(args.model_type))
    model.save(model_file)
    print('Model saved as {}'.format(model_file))


if args.summary:
    # Report the model architecture, number of params, etc.
    model.summary()

    # Report memory usage, in bytes.
    print('Model memory (bytes): {}'.format(get_model_memory_usage(model, args.batch_size)))

    # FIXME: For computing FLOPs we must pass in a model file.
    if args.eval_model:
        print('Model FLOPS: {}'.format(get_model_flops(args.eval_model)))


# At this point we have a trained model, be it loaded from a previous experiment or just trained right before,
# so we're ready to evaluate its classification performance.
y_pred = model.predict(X_test).ravel()

# Compute evaluation metrics, see `utils.py`.
res = eval_binary_classifier(y_test, y_pred)

precision, recall, f1, _ = res['prf_binary']
adj_precision, adj_recall, adj_f1, _ = res['prf_weighted']
accuracy, auc = res['acc'], res['auc_macro']

print('''
| Mode         | Precision | Recall | F-measure | Accuracy | AUC ROC |
|---           |---        |---     |---        |---       |---      |
| non-weigthed | {:.4f}    | {:.4f} | {:.4f}    | {:.4f}   | {:.4f}  |
| weigthed     | {:.4f}    | {:.4f} | {:.4f}    | {:.4f}   | {:.4f}  |
'''.format(precision, recall, f1, accuracy, auc,
           adj_precision, adj_recall, adj_f1, accuracy, auc))

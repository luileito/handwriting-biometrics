#!/usr/bin/env python3

'''
Luis A. Leiva, Moises Diaz, Miguel A. Ferrer, Réjean Plamondon.
Human or Machine? It Is Not What You Write, But How You Write It.
Proc. ICPR, 2020.

Helper functions.

Author: Luis A. Leiva <luis.leiva@aalto.fi>
Last modified: 03.09.2020
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.python.keras import backend as K


def recall(y_true, y_pred):
    '''Compute Recall score.'''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    '''Compute Precision score.'''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    '''Compute F1 score.'''
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec*rec) / (prec + rec + K.epsilon())


def eval_binary_classifier(y_true, y_pred, class_weights=None):
    # NB: `y_true = [1, 0, 0, ...]` but `y_pred = [[0.2324], [0.8731], ...]`.
    # Remember we're dealing with 1 output neuron with sigmoid activation.
    prob_labels = np.array(y_pred).ravel()
    pred_labels = prob_labels > 0.5

    true_labels = y_true.astype(int)
    pred_labels = pred_labels.astype(int)

    return {
        'acc': accuracy_score(true_labels, pred_labels),
        'prf_weighted': precision_recall_fscore_support(true_labels, pred_labels, average='weighted'),
        'prf_binary': precision_recall_fscore_support(true_labels, pred_labels, average='binary'),
        'prf_micro': precision_recall_fscore_support(true_labels, pred_labels, average='micro'),
        'prf_macro': precision_recall_fscore_support(true_labels, pred_labels, average='macro'),
        'auc_weighted': roc_auc_score(y_true, y_pred, average='weighted'),
        'auc_micro': roc_auc_score(y_true, y_pred, average='micro'),
        'auc_macro': roc_auc_score(y_true, y_pred, average='macro'),
        'conf_matrix': confusion_matrix(true_labels, pred_labels)
    }


def get_model_memory_usage(model, batch_size):
    '''Compute model memory cost in MB.'''
    shapes_mem_count = 0
    internal_model_mem_count = 0

    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)

        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    float_prec = K.floatx()
    if float_prec == 'float16':
        number_size = 2.0
    elif float_prec == 'float32':
        number_size = 4.0
    else: # float64
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)

    return np.round(total_memory + internal_model_mem_count)


def get_model_flops(model_file):
    '''Compute model memory cost in FLOPs.'''
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        with session.as_default():
            # Now that the session is initialized, we can load the model.
            # NB: The model should be compiled to get an accurate flops estimation.
            tf.keras.models.load_model(model_file)

            meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            comp = tf.compat.v1.profiler.profile(graph=graph, run_meta=meta, cmd='op', options=opts)

    # Reset state to avoid accumulating computations,
    # in case we call this function more than once.
    tf.compat.v1.reset_default_graph()

    return comp.total_float_ops

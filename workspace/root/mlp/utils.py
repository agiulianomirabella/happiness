from root.parameters import STATS_COLUMNS

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers # pylint: disable= import-error
from tensorflow.keras.layers import Dense, Dropout # pylint: disable= import-error
from tensorflow.keras.models import Sequential # pylint: disable= import-error

def print_info(training_info, folder):
    print('\n\nService: {}, Model: {}. K training: {}/{}\n\n'.format(training_info['service'], training_info['model_name'], folder+1, training_info['k']))

def read_architecture(architecture_name):
    with open('root/mlp/architectures.json') as json_file:
        return json.load(json_file)[architecture_name]

def compute_auto_name(path):
    unavailables = os.listdir(path)
    i = 0
    while 'auto'+str(i) in unavailables:
        i = i + 1
    os.mkdir(path + 'auto'+str(i))
    return 'auto'+str(i)

def get_model(input_dim, architecture_name, optimizer, learning_rate, momentum):

    model = Sequential()
    layers = read_architecture(architecture_name)
    first_layer = layers.pop(0)

    model.add(Dense(first_layer['units'], activation=first_layer['activation'], kernel_regularizer= regularizers.l1_l2(l1=first_layer['l1'], l2=first_layer['l2']), input_shape=(input_dim, )))

    for layer in layers:
        model.add(Dense(layer['units'], activation=layer['activation'], kernel_regularizer= regularizers.l1_l2(l1=layer['l1'], l2=layer['l2'])))
        if 'dropout' in layer.keys():
            model.add(Dropout(layer['dropout']))

    model.add(Dense(1,  activation='relu'))

    if optimizer.lower()=='sgd':
        o = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    else:
        o = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=o, loss='mse', metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError()])
    model.summary()

    return model

def append_to_stats(stats, file_path):
    if not os.path.exists(file_path):
        stats_df = pd.DataFrame(columns= STATS_COLUMNS)
    else:
        stats_df = pd.read_csv(file_path)

    stats_df = stats_df.append(stats, ignore_index=True)
    stats_df.to_csv(file_path, index=False)


def compute_training_scores(histories):
    metrics = list(histories[0].keys())
    max_epoch = max([len(h[metrics[0]]) for h in histories])
    for h in histories:
        for m in metrics:
            h[m] = pad_history(h[m], max_epoch)

    out= {}
    mean_per_epoch = {}
    training_scores = {}
    
    for m in metrics:
        mean = []
        for epoch in range(max_epoch):
            for h in histories:
                mean.append(h[m][epoch])
        mean_per_epoch[m] = np.mean(mean)

        training_scores[m]               = np.min(np.mean(mean))
        training_scores[m + '_epoch']    = np.argmin(np.mean(mean))

    for m in metrics:
        name = 'train_' + m.replace('mean_squared_error', 'mse').replace('mean_absolute_percentage_error', 'mape')
        out[name.replace('train_val_', 'valid_')] = training_scores[m]

    return out

def pad_history(l, width):
    content = l[-1]
    l.extend([content] * (width - len(l)))
    return l

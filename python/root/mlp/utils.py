import json
import os
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

import json
import tensorflow as tf
from tensorflow.keras import regularizers # pylint: disable= import-error
from tensorflow.keras.layers import Dense, Dropout # pylint: disable= import-error
from tensorflow.keras.models import Sequential # pylint: disable= import-error

def print_info(training_info, folder):
    print('\n\nService: {}, Model: {}. K training: {}/{}\n\n'.format(training_info['service'], training_info['model_name'], folder+1, training_info['k']))

def read_architecture(architecture_name):
    with open('root/mlp/architectures.json') as json_file:
        return json.load(json_file)[architecture_name]

def get_model(input_dim, architecture_name, optimizer):

    model = Sequential()
    layers = read_architecture(architecture_name)
    first_layer = layers.pop(0)

    model.add(Dense(first_layer['units'], activation=first_layer['activation'], kernel_regularizer= regularizers.l1_l2(l1=first_layer['l1'], l2=first_layer['l2']), input_shape=(input_dim, )))

    for layer in layers:
        model.add(Dense(layer['units'], activation=layer['activation'], kernel_regularizer= regularizers.l1_l2(l1=layer['l1'], l2=layer['l2'])))
        if 'dropout' in layer.keys():
            model.add(Dropout(layer['dropout']))

    model.add(Dense(1,  activation='relu'))

    model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
    model.summary()

    return model

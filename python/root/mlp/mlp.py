from root.mlp.parameters import BATCH_SIZE, EPOCHS, METRICS, K, MLP_PATH, TARGET, SERVICES
from root.mlp.preprocessing import train_preprocessing, train_test_preprocessing
from root.mlp.models import get_model
from root.mlp.utils import check_service, make_folders, get_model_info, print_info, make_auto_folder
from root.mlp.stats import append_to_stats
from root.mlp.scores import compute_training_scores, best_model
from root.mlp.plots import plot_architecture, plot_best_model_scores
from root.data.data import get_df, get_train_test_df

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

def kfold(service, df = None, architecture='standard0', stats_file='default.csv', batch_size= BATCH_SIZE, optimizer= 'Adam', 
            learning_rate= 0.01, momentum= 0, epochs= EPOCHS, k= K, metrics = METRICS, 
            plot_model= False):

    if not isinstance(df, pd.DataFrame):
        print('*** WARNING: missing dataframe. default is get_df(service)')
        df = get_df(service)

    check_service(service)
    make_folders()
    df = train_preprocessing(df, service)
    model_name = architecture + '_B' + str(batch_size) + '_O' + optimizer + '_L' + str(learning_rate) + '_M' + str(momentum)
    info = get_model_info(model_name)
    info.update({
        'service':        service,
        'mode':           'train',
        'n_columns':      len(df.columns)-1,
        'epochs':         epochs,
        'k': k
    })

    histories = []

    skf = StratifiedKFold(n_splits = k, shuffle = True)
    folder = 0

    for train_index, valid_index in skf.split(np.zeros(len(df.index)), df[[TARGET]]):

        print_info(info, folder)

        train_df = df.iloc[train_index].copy()
        valid_df = df.iloc[valid_index].copy()

        info.update({
            'n_train':        len(train_df.index),
            'n_valid':        len(valid_df.index)
        })

        train_labels=train_df.pop(TARGET)
        valid_labels=valid_df.pop(TARGET)

        model = get_model(info)

        if plot_model:
            plot_architecture(info, model)
            plot_model = False

        history = model.fit(
            x = train_df.values,
            y = train_labels.values,
            validation_data=(valid_df.values, valid_labels.values),
            batch_size = batch_size,
            epochs = epochs,
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, mode='auto', restore_best_weights=True
                )
            ]
        )

        histories.append(history.history)
        folder = folder + 1

    info.update(compute_training_scores(histories, metrics))
    append_to_stats(info, stats_file)

    return info

def holdout(service, train_df=None, test_df=None, architecture='standard0', stats_file='default.csv', 
            batch_size= BATCH_SIZE, optimizer= 'Adam', 
            learning_rate= 0.01, momentum= 0, epochs= EPOCHS, metrics = METRICS, 
            plot_model= False, callbacks = ['early_stopping'], patience= 5, monitor= 'val_loss'):

    if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame):
        print('*** WARNING: missing dataframes. default is get_train_test_df(service), chosen randomly')
        train_df, test_df = get_train_test_df(service)

    make_folders()
    train_df, test_df = train_test_preprocessing(train_df, test_df, service)
    
    if momentum == 0:
        momentum = '0.0'
    model_name = architecture + '_B' + str(batch_size) + '_O' + optimizer + '_L' + str(learning_rate) + '_M' + str(momentum)

    valid_df = train_df.sample(frac=0.2)
    train_df = train_df.drop(valid_df.index)
    valid_labels = valid_df.pop(TARGET)
    train_labels = train_df.pop(TARGET)
    test_labels  =  test_df.pop(TARGET)

    info = get_model_info(model_name)
    info.update({
        'service':        service,
        'mode':           'test',
        'n_train':        len(train_df.index),
        'n_valid':        len(valid_df.index),
        'n_test':         len(test_df.index),
        'n_columns':      len(train_df.columns),
        'epochs':         epochs,
        'monitors':       monitor,
        'callbacks':      "&".join(callbacks),
        'patience':       patience
    })

    model = get_model(info)

    model.fit(
        x = train_df.values,
        y = train_labels.values,
        validation_data=(valid_df.values, valid_labels.values),
        batch_size = batch_size,
        epochs = epochs,
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor, patience=patience, mode='auto', restore_best_weights=True
            )
        ]
    )

    scores = model.evaluate(test_df.values, test_labels.values, batch_size=info['batch_size'])
    stats = {
        'test_auc':      scores[2],
        'test_accuracy': scores[1],
        'test_loss':     scores[0]
    }

    info.update(stats)
    append_to_stats(info, stats_file)

    return scores

def auto_best(stats_file='default.csv', architectures = ['standard0'], batch_sizes= [8], optimizers= ['Adam'], 
            learning_rates= [0.001], epochs= EPOCHS, plot_scores= True):
    make_folders()
    new_save_path = make_auto_folder()

    print('Number of experiments per service: {}'.format(len(architectures)*len(batch_sizes)*len(optimizers)*len(learning_rates)))

    for s in SERVICES:
        train_df, test_df = get_train_test_df(s, test_frac= 0.2)
        train_df.to_csv(new_save_path + s + '_train.csv', index=False)
        test_df.to_csv( new_save_path + s + '_test.csv',  index=False)
        for a in architectures:
            for b in batch_sizes:
                for o in optimizers:
                    for l in learning_rates:
                        kfold(s, df=train_df, architecture=a, batch_size= b, optimizer= o, learning_rate= l, epochs= epochs, stats_file=stats_file)
    
    best_model_name = best_model(stats_file, 'valid')
    best_model_info = get_model_info(best_model_name)

    stats = []

    for s in SERVICES:
        train_df = pd.read_csv(new_save_path + s + '_train.csv')
        test_df  = pd.read_csv(new_save_path + s + '_test.csv')
        holdout(s, train_df=train_df, test_df=test_df, stats_file = stats_file,
            architecture= best_model_info['architecture'], batch_size= best_model_info['batch_size'], 
            optimizer= best_model_info['optimizer'], learning_rate= best_model_info['learning_rate'], 
            momentum= best_model_info['momentum'], epochs= EPOCHS, 
            plot_model= True, callbacks = ['early_stopping'], patience= 5)

    if plot_scores:
        plot_best_model_scores(stats_file)

    return best_model_name

def accuracy_evolution_data_size(save_name= 'accuracy_evolution.csv', sizes= None):
    df = get_df(SERVICES[0])
    if not isinstance(sizes, list):
        sizes = list(np.linspace(0, len(df.index)*2/3, num=20, dtype=int))[1:]
        print('***WARNING: no sizes given, {} is used'.format(sizes))

    for s in sizes:
        for service in SERVICES:
            df = get_df(service).sample(n= s)
            kfold(service, df = df, stats_file= save_name[:-4] + '_aux.csv')

    evolution = pd.read_csv('../data/mlp/stats/' + save_name[:-4] + '_aux.csv')
    out = pd.DataFrame([
            {
                'n': n, 
                'accuracy_mean': np.mean(evolution[evolution['n_train']==n]['valid_accuracy']), 
                'accuracy_std':  np.std(evolution[evolution['n_train']==n]['valid_accuracy'])
            }
            for n in list(np.sort(list(set(evolution['n_train'].values))))
        ])
    out.to_csv('../data/mlp/stats/'+save_name, index=False)

    return out



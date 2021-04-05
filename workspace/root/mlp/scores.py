import numpy as np
import pandas as pd

def get_model_scores(path, name):
    test_stats          = pd.read_csv(path + 'test_scores.csv')
    best_model_test_row = test_stats[test_stats['name'] == name].iloc[0]

    scores = {
        'test_loss':    best_model_test_row['test_loss'],
        'test_mse':     best_model_test_row['test_mse'],
        'test_mape':    best_model_test_row['test_mape']
    }

    return scores

def best_model(stats_file):
    train_stats = pd.read_csv(stats_file)

    best_model_train_row = train_stats.iloc[train_stats['valid_mse'].idxmin()]

    best_model_info = {
        'name':              best_model_train_row['name'],
        'architecture_name': best_model_train_row['architecture'],
        'batch_size':        best_model_train_row['batch_size'],
        'optimizer':         best_model_train_row['optimizer'],
        'learning_rate':     best_model_train_row['learning_rate'],
        'momentum':          best_model_train_row['momentum']
    }

    return best_model_info


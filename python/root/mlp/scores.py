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

def best_model(path):
    train_stats = pd.read_csv(path + 'train_scores.csv')

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

import numpy as np
import pandas as pd

def best_model(stats_file):
    stats = pd.read_csv(stats_file)
    best_model = stats.iloc[stats['valid_mse'].idxmin()]

    out = {
        'architecture_name': best_model['architecture'],
        'batch_size':        best_model['batch_size'],
        'optimizer':         best_model['optimizer'],
        'learning_rate':     best_model['learning_rate'],
        'momentum':          best_model['momentum']
    }

    return out

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

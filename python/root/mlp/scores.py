import numpy as np
import pandas as pd

def best_model(stats_file, mode):
    stats = pd.read_csv(STATS_PATH + stats_file)
    stats = stats[stats[mode + '_accuracy'].notna()]
    accuracy_scores = {np.mean(stats[stats['model_name'] == m][mode+'_accuracy']): m for m in set(stats['model_name'].values)}
    best_model = accuracy_scores[max(accuracy_scores.keys())]
    print('The best model is: {}'.format(best_model))
    return best_model

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

        if m=='loss' or m=='val_loss':
            training_scores[m]              = np.min(mean_per_epoch[m])
            training_scores[m + '_epoch']    = np.argmin(mean_per_epoch[m])
        else:
            training_scores[m]              = np.max(mean_per_epoch[m])
            training_scores[m + '_epoch']    = np.argmax(mean_per_epoch[m])

    for m in metrics:
        if m[:4] == 'val_':
            out[m.replace('val_', 'valid_').replace('_mean_squared_error', '_mse')] = training_scores[m]
        else:
            out['train_' + m.replace('mean_squared_error', 'mse')] = training_scores[m]

    return out

def pad_history(l, width):
    content = l[-1]
    l.extend([content] * (width - len(l)))
    return l

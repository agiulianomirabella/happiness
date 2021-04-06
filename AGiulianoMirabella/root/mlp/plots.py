import pandas as pd
from root.utils import pretty_dict
from root.mlp.scores import best_model

import matplotlib.pyplot as plt

def plot_model_scores_per_year(name, stats_file='../data/mlp/default.csv'):
    stats = pd.read_csv(stats_file)
    stats = stats[stats['year']!=pd.NA]
    if not name in stats['name'].values:
        print('ERROR: there is no record for the model {}'.format(name))
        return None
    stats = stats[stats['name']==name]
    stats = stats.groupby(['year']).agg(score = ('test_mape', 'mean'))
    stats = pd.DataFrame(stats.to_records())
    years, scores = stats['year'].values, stats['score'].values
    plt.plot(years, scores)
    plt.savefig('../data/mlp/'+ name +'_scores_per_year.png')
    plt.clf()

def plot_best_model_scores_per_year(stats_file='../data/mlp/default.csv'):
    name = best_model(stats_file)['name']
    plot_model_scores_per_year(name, stats_file)

def print_best_model_scores(stats_file='../data/mlp/default.csv'):
    out = best_model(stats_file)
    pretty_dict(out)



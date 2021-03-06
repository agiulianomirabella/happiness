from pandas.core.indexes.base import Index
from root.mlp.plots import plot_best_model_scores_per_year, plot_model_scores_per_year
from root.utils import pretty_dict
from root.mlp.regressor import Regressor
from root.parameters import BATCH_SIZE, EPOCHS, K
from root.mlp.utils import append_to_stats, compute_auto_name
from root.mlp.scores import best_model, get_model_scores

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

class AutoRegressor:
    def __init__(self, 
        preprocessed_data,
        epochs = EPOCHS,
        k = K,
        architecture_names=['arch0'], 
        batch_sizes=[BATCH_SIZE], 
        optimizers=['sgd'], 
        learning_rates=[0.1], 
        momentums=[0], 
        path='../data/mlp/auto/'):

        self.data = preprocessed_data
        self.epochs = epochs
        self.k = k
        self.path = path + compute_auto_name(path) + '/'

        self.info = {
            'architectures': architecture_names,
            'batch_sizes': batch_sizes,
            'optimizers': optimizers,
            'learning_rates': learning_rates,
            'momentums': momentums
        }
        self.architectures       = architecture_names
        self.batch_sizes         = batch_sizes
        self.optimizers          = optimizers
        self.learning_rates      = learning_rates
        self.momentums           = momentums

    def print_scores(self):
        best_model_info = best_model(self.path  + 'train_scores.csv')
        scores = get_model_scores(self.path, best_model_info['name'])
        print('\nTotal number of experiments: {}'.format(len(self.architectures)*len(self.batch_sizes)*len(self.optimizers)*len(self.learning_rates)*len(self.momentums)))
        print('The best model is:')
        pretty_dict(best_model_info)
        print('The scores obtained:')
        pretty_dict(scores)

    def get_scores(self):
        best_model_info = best_model(self.path  + 'train_scores.csv')
        scores = get_model_scores(self.path, best_model_info['name'])
        return scores

    def best(self):
        train_df, test_df = train_test_split(self.data, test_size=0.2)
        train_df.to_csv(self.path + 'train.csv', index=False)
        test_df.to_csv( self.path + 'test.csv', index=False)
        for a in self.architectures:
            for b in self.batch_sizes:
                for l in self.learning_rates:
                    for o in self.optimizers:
                        if o.lower == 'adam':
                            regressor = Regressor(
                                len(train_df.columns)-1, 
                                architecture_name=a, 
                                batch_size = b, 
                                optimizer= o, 
                                learning_rate= l, 
                                stats_file= 'train_scores.csv', 
                                path= self.path)
                            regressor.kfold(train_df, epochs = self.epochs, k = self.k)
                        else:
                            for m in self.momentums:
                                regressor = Regressor(
                                    len(train_df.columns)-1, 
                                    architecture_name=a, 
                                    batch_size = b, 
                                    optimizer= o, 
                                    learning_rate= l, 
                                    momentum= m, 
                                    stats_file= 'train_scores.csv', 
                                    path= self.path)
                                regressor.kfold(train_df, epochs = self.epochs, k = self.k)
        
        best_model_info = best_model(self.path + 'train_scores.csv')
        self.model_info = best_model_info
        best_regressor = Regressor(
            len(train_df.columns)-1,
            architecture_name = best_model_info['architecture_name'],
            batch_size=best_model_info['batch_size'],
            optimizer= best_model_info['optimizer'],
            learning_rate= best_model_info['learning_rate'],
            momentum= best_model_info['momentum'],
            stats_file = 'test_scores.csv',
            path= self.path
        )

        train_df = pd.read_csv(self.path + 'train.csv')
        test_df  = pd.read_csv(self.path + 'test.csv')

        stats = best_regressor.holdout(train_df, test_df)
        append_to_stats(pd.DataFrame([stats]), self.path+'stats.csv')

        return stats

    def plot(self):
        plot_model_scores_per_year(self.model_info['name'], self.path+'stats.csv')


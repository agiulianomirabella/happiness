from root.utils.utils import delete_file
from root.utils.mlp.utils import check_csv
from root.utils.mlp.scores import best_model
from root.utils.parameters import SERVICES, STATS_PATH, PLOTS_PATH, ARCHITECTURES_PATH

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

def plot_best_model_scores(stats_file):
    for mode in ['train', 'valid', 'test']:
        info = {
            'model_name': best_model(stats_file, 'valid'),
            'mode': mode
        }
        plot_model_scores(stats_file, info, 'best_model_'+stats_file+'_'+mode)

def plot_model_scores(stats_file, mode, model_name):
    save_name = model_name + '_scores_on_' + mode + '_stats_file'[:-4]
    stats_df = pd.read_csv(STATS_PATH + stats_file)
    stats_df = stats_df[stats_df['model_name'] == model_name]
    stats_df = stats_df[stats_df[mode+'_accuracy'].notna()]

    acc_values = [stats_df[stats_df['service'] == s][mode + '_accuracy'].mean() for s in SERVICES]
    AUC_values = [stats_df[stats_df['service'] == s][mode + '_auc'].mean() for s in SERVICES]
    labels = SERVICES
    
    x = np.arange(len(labels)) # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize = (15, 5))
    rects1 = ax.bar(x - width/2, AUC_values, width, label='AUC', color='lightgray')
    rects2 = ax.bar(x + width/2, acc_values, width, label='accuracy', color='steelblue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title(model_name + ' scores on ' + mode + ' data', pad=20)
    ax.set_xticks(x)
    plt.xticks(rotation=45)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0, top=1)
    ax.legend(loc='lower right')

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    fig.tight_layout()
    fig.savefig(PLOTS_PATH + save_name + '.png', bbox_inches='tight')
    print(model_name + ' scores:')
    print('Accuracy:')
    print('   Mean: {}'.format(np.mean(acc_values)))
    print('   Std:  {}'.format(np.std(acc_values)))
    print('AUC:')
    print('   Mean: {}'.format(np.mean(AUC_values)))
    print('   Std:  {}'.format(np.std(AUC_values)))
    plt.clf()

def plot_services_scores(stats_file, info, save_name, show_models= False):
    mode = info['mode']
    check_csv(stats_file)

    stats_df = pd.read_csv(STATS_PATH + stats_file)

    acc_values = [stats_df[stats_df['service'] == s][mode + '_accuracy'].max() for s in SERVICES]
    AUC_values = [stats_df['test_auc'].iloc[stats_df[stats_df['service'] == s][mode + '_accuracy'].idxmax()] for s in SERVICES]
    models     = [stats_df['model_name'].iloc[stats_df[stats_df['service'] == s][mode + '_accuracy'].idxmax()] for s in SERVICES]

    if show_models:
        labels = [s + ': ' + models[i] for i, s in enumerate(SERVICES)]
    else:
        labels = SERVICES

    x = np.arange(len(labels)) # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize = (15, 5))
    rects1 = ax.bar(x - width/2, AUC_values, width, label='AUC', color='lightgray')
    rects2 = ax.bar(x + width/2, acc_values, width, label='accuracy', color='steelblue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Best scores by service', pad=20)
    ax.set_xticks(x)
    plt.xticks(rotation=45)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0, top=1)
    ax.legend(loc='lower right')

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    fig.tight_layout()    
    fig.savefig(PLOTS_PATH + save_name + '.png', bbox_inches='tight')
    plt.clf()

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = round(rect.get_height(), 3)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize='small')
    
def plot_architecture(info, model):
    tf.keras.utils.plot_model(model, to_file=ARCHITECTURES_PATH+info['service']+'_'+info['architecture']+'.png', dpi=72, rankdir="LR", show_shapes=True)


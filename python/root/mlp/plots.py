from root.utils import pretty_dict
from root.mlp.scores import best_model

def print_best_model_scores(stats_file):
    out = best_model(stats_file)
    pretty_dict(out)



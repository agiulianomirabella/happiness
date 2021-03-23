from root.parameters import STATS_COLUMNS

import os
import pandas as pd

def append_to_stats(stats, path, stats_csv_filename):
    if not os.path.exists(path + stats_csv_filename):
        stats_df = pd.DataFrame(columns= STATS_COLUMNS)
    else:
        stats_df = pd.read_csv(path + stats_csv_filename)

    stats_df = stats_df.append(stats, ignore_index=True)
    stats_df.to_csv(path + stats_csv_filename, index=False)

from root.parameters import STATS_COLUMNS

import os
import pandas as pd

def append_to_stats(stats, file_path):
    if not os.path.exists(file_path):
        stats_df = pd.DataFrame(columns= STATS_COLUMNS)
    else:
        stats_df = pd.read_csv(file_path)

    stats_df = stats_df.append(stats, ignore_index=True)
    stats_df.to_csv(file_path, index=False)

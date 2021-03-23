from root.utils import pretty_dict
from root.preprocessing import regressor_preprocess
from root.read_data import read_data
from root.mlp.classifier import Regressor
from sklearn.model_selection import train_test_split

raw_data = read_data()

preprocessed_data = regressor_preprocess(raw_data).iloc[:50]

regressor = Regressor(len(list(preprocessed_data.columns)), 'arch0')
regressor.plot_model()

# score = regressor.kfold(preprocessed_data)


train_df, test_df = train_test_split(preprocessed_data, test_size=0.2)
score = regressor.holdout(train_df, test_df)


pretty_dict(score)


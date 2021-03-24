from root.mlp.auto import AutoRegressor
from root.preprocessing import regressor_preprocess
from root.read_data import read_data

raw_data = read_data()

preprocessed_data = regressor_preprocess(raw_data)

batch_sizes    = [4, 8, 16, 32, 64]
learning_rates = [0.01, 0.05, 0.1, 0.2]
momentums      = [0, 0.01, 0.05, 0.1, 0.2]
optimizers     = ['Adam', 'sgd']

batch_sizes    = [32]
learning_rates = [0.1]
momentums      = [0]
optimizers     = ['sgd']

auto = AutoRegressor(
    preprocessed_data,
    batch_sizes=    batch_sizes,
    learning_rates= learning_rates,
    momentums=      momentums,
    optimizers=     optimizers
)

auto.best()
auto.get_score()



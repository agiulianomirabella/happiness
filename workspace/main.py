from root.read_data import read_data
from root.mlp.auto import AutoRegressor
from root.preprocessing import regressor_preprocess

raw_data = read_data()
preprocessed_data = regressor_preprocess(raw_data)

architectures  = ['arch0', 'arch1']
batch_sizes    = [4, 8, 16]
learning_rates = [0.01, 0.1]
momentums      = [0]
optimizers     = ['Adam', 'sgd']

auto = AutoRegressor(
    preprocessed_data,
    architecture_names = architectures,
    batch_sizes =        batch_sizes,
    learning_rates =     learning_rates,
    momentums =          momentums,
    optimizers =         optimizers
)

auto.best()
auto.get_score()
auto.plot()


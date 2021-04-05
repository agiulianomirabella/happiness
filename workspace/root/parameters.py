
YEARS = [2015, 2016, 2017, 2018, 2019]
TARGET = 'score'

STATS_COLUMNS = [
    'year',
    'name',
    'architecture',
    'batch_size',
    'epochs',
    'optimizer',
    'learning_rate',
    'momentum',
    'mode',
    'k',
    'n_train',
    'n_test',
    'n_columns',

    'train_mse',
    'valid_mse',
    'test_mse',

    'train_mape',
    'valid_mape',
    'test_mape',

    'train_loss',
    'valid_loss',
    'test_loss'

]

##### DEFAULT HYPERPARAMETERS:
BATCH_SIZE  = 32
EPOCHS      = 100
K           = 5

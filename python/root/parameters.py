
TARGET = 'score'

STATS_COLUMNS = [
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

    'train_loss',
    'valid_loss',
    'test_loss'

]

##### DEFAULT HYPERPARAMETERS:
BATCH_SIZE  = 8
EPOCHS      = 10
K           = 3

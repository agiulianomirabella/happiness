from root.parameters import BATCH_SIZE, EPOCHS, K, TARGET
from root.mlp.utils import get_model
from root.mlp.scores import compute_training_scores
from root.mlp.stats import append_to_stats

import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split

class Regressor:
    def __init__(self, input_dim, architecture_name='arch0', batch_size=BATCH_SIZE, optimizer='sgd', learning_rate=0.1, momentum=0, stats_file = 'default.csv', path='../data/mlp/'):
        self.path = path
        self.name = architecture_name + '_B' + str(batch_size) + '_O' + optimizer + '_L' + str(learning_rate) + '_M' + str(momentum)

        self.info = {
            'name': self.name,
            'architecture': architecture_name,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'momentum': momentum
        }
        self.architecture_name  = architecture_name
        self.batch_size         = batch_size
        self.optimizer          = optimizer
        self.learning_rate      = learning_rate
        self.momentum           = momentum

        self.stats_file = stats_file
        self.model = get_model(input_dim, architecture_name, optimizer)

    def plot_model(self):
        tf.keras.utils.plot_model(self.model, self.path + self.name + '_architecture.png', dpi=72, rankdir="LR", show_shapes=True)

    def kfold(self, preprocessed_data, epochs= EPOCHS, k= K):

        histories = []

        y = preprocessed_data.copy().pop(TARGET).values
        X = preprocessed_data.values

        kf = KFold(n_splits=k, shuffle=True)

        for train_index, test_index in kf.split(X):
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]

            history = self.model.fit(
                x = X_train,
                y = y_train,
                validation_data=(X_valid, y_valid),
                batch_size = self.batch_size,
                epochs = epochs,
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=5, mode='auto', restore_best_weights=True
                    )
                ]
            )
            histories.append(history.history)

        stats = compute_training_scores(histories)
        stats.update(self.info)
        stats.update({
            'k': k,
            'epochs': epochs,
            'mode': 'kfold',
            'n_train': len(preprocessed_data.index),
            'n_columns': len(preprocessed_data.columns) - 1,
        })

        append_to_stats(stats, self.path, self.stats_file)

        return stats

    def holdout(self, train_df, test_df, epochs=EPOCHS):

        y_train = train_df.copy().pop(TARGET).values
        y_test  =  test_df.copy().pop(TARGET).values

        X_train = train_df.values
        X_test  =  test_df.values

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

        self.model.fit(
            x = X_train,
            y = y_train,
            validation_data=(X_valid, y_valid),
            batch_size = self.batch_size,
            epochs = epochs,
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, mode='auto', restore_best_weights=True
                )
            ]
        )

        evaluation = self.model.evaluate(X_test, y_test, batch_size=self.batch_size)
        stats = {
            'mode':        'holdout',
            'n_train':     len(train_df.index),
            'n_test':      len(test_df.index),
            'n_columns':   len(train_df.columns) - 1,
            'epochs':      epochs,
            'test_mse':    evaluation[0],
            'test_loss':   evaluation[1]
        }
        stats.update(self.info)

        append_to_stats(stats, self.path, self.stats_file)
        print(self.model.metrics_names)

        return stats


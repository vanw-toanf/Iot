import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


class Tiny_model():
    def __init__(self):
        self.model = Sequential([
            Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(96, 96, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(16, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(9, activation='relu')
        ])

        self.callbacks = []

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[ Precision(), Recall(), F1Score() ]
        )

    def fit(self, X_train, y_train, validation_data, callbacks, batch_size=32, epochs=100):
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks
        )
        return history

    def save_model(self, path):
        self.model.save(path)
# Tiny_model = Sequential([
#     Conv2D( 8, kernel_size=(3, 3), activation='relu', input_shape=(96, 96, 3)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(16, kernel_size=(3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(32, kernel_size=(3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(9, activation='sigmoid')
# ])
#
# Tiny_model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=[ Precision(), Recall(), F1Score() ]
# )
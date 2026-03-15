
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier

# Set seeds 
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# MLP MODEL DEFINITION
def build_mlp(meta):
    n_features = meta["n_features_in_"]

    model = Sequential([
        Input(shape=(n_features,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


#  SAFE WRAPPER ( for 2 dimension  output (n_samples, 2)) this is important to fix single sigmoid output error
class SafeKerasClassifier(KerasClassifier):
    def predict_proba(self, X, **kwargs):
        proba = super().predict_proba(X, **kwargs)

        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        if proba.shape[1] == 1:
            proba = np.hstack([1 - proba, proba])

        return proba

# Instantiate the MLP classifier
mlp_clf = SafeKerasClassifier(
    model=build_mlp,
    epochs=1,
    batch_size=32,
    verbose=0,
    validation_split=0.1,
    random_state=42          
)


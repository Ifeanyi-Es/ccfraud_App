
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def build_mlp(meta):
    n_features = meta["n_features_in_"]

    model = Sequential([
        Input(shape=(n_features,)),               # Input layer
        Dense(64, activation='relu'),             # Hidden Layer 1 
        Dropout(0.2),
        Dense(32, activation='relu'),             # Hidden Layer 2 
        Dense(1, activation='sigmoid')            # Output layer
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model



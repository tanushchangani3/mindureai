import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def build_autoencoder(input_dim=100, encoding_dim=32):
    # Define the input layer
    input_layer = Input(shape=(input_dim,))
    
    # Define the encoding layer
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    
    # Define the decoding layer
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    # Build and compile the autoencoder model
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

if __name__ == "__main__":
    # Build and summarize the autoencoder model
    autoencoder = build_autoencoder()
    autoencoder.summary()
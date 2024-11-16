import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

def build_lstm_model(timesteps=10, input_dim=5):
    # Define the input layer
    input_layer = Input(shape=(timesteps, input_dim))
    
    # Define the LSTM layer
    lstm = LSTM(50, return_sequences=False)(input_layer)
    
    # Define the output layer
    output = Dense(1, activation='linear')(lstm)
    
    # Build and compile the LSTM model
    model = Model(input_layer, output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    # Build and summarize the LSTM model
    model = build_lstm_model()
    model.summary()
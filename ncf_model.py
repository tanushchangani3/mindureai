# Neural Collaborative Filtering (NCF)
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def build_ncf_model(num_users, num_items, latent_dim=50):
    user_input = Input(shape=(1,))
    item_input = Input(shape=(1,))
    user_embedding = Embedding(input_dim=num_users, output_dim=latent_dim)(user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=latent_dim)(item_input)
    user_vec = Flatten()(user_embedding)
    item_vec = Flatten()(item_embedding)
    concat = Concatenate()([user_vec, item_vec])
    dense = Dense(128, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(dense)
    model = Model([user_input, item_input], output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = build_ncf_model(num_users=1000, num_items=500)
    model.summary()
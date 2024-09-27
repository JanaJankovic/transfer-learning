import numpy as np
import tensorflow as tf
import optuna
import pandas as pd
import utils.tools as tls
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import utils.preprocessing as pp
import utils.evaluation as eval
import utils.tools as tls
import utils.constants as c


def build_rnn_model(input_shape, num_layers, num_neurons, learning_rate, dropout_rate):
    model = Sequential()
    
    # First RNN layer, return full sequence for stacking more layers
    model.add(SimpleRNN(num_neurons, activation='relu', return_sequences=(num_layers > 1), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Additional RNN layers (if any)
    for i in range(1, num_layers):
        # Only the last layer should not return sequences
        return_sequences = (i < num_layers - 1)
        model.add(SimpleRNN(num_neurons, activation='relu', return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))
        
    # Output layer: single unit for regression (predicting a single value)
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

def build_lstm_model(input_shape, num_layers, num_neurons, learning_rate, dropout_rate):
    model = Sequential()
    
    # First RNN layer, return full sequence for stacking more layers
    model.add(LSTM(num_neurons, activation='relu', return_sequences=(num_layers > 1), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Additional RNN layers (if any)
    for i in range(1, num_layers):
        # Only the last layer should not return sequences
        return_sequences = (i < num_layers - 1)
        model.add(LSTM(num_neurons, activation='relu', return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))
        
    # Output layer: single unit for regression (predicting a single value)
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    
    return model

def build_gru_model(input_shape, num_layers, num_neurons, learning_rate, dropout_rate):
    model = Sequential()
    
    # First RNN layer, return full sequence for stacking more layers
    model.add(GRU(num_neurons, activation='relu', return_sequences=(num_layers > 1), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Additional RNN layers (if any)
    for i in range(1, num_layers):
        # Only the last layer should not return sequences
        return_sequences = (i < num_layers - 1)
        model.add(GRU(num_neurons, activation='relu', return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))
        
    # Output layer: single unit for regression (predicting a single value)
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    
    return model


def build_model(best_params, input_shape):
    if best_params['network_type'] == "RNN":
        return build_rnn_model(input_shape, best_params['num_layers'], best_params['num_neurons'], best_params['learning_rate'], best_params['dropout_rate'])
    elif best_params['network_type'] == "LSTM":
        return build_lstm_model(input_shape, best_params['num_layers'], best_params['num_neurons'], best_params['learning_rate'], best_params['dropout_rate'])
    elif best_params['network_type'] == "GRU":
        return build_gru_model(input_shape, best_params['num_layers'], best_params['num_neurons'], best_params['learning_rate'], best_params['dropout_rate'])
    
    
def add_simple_RNN_layers(input_layer, num_layers, neurons_per_layer, dropout_rate):
    x = input_layer
    for i in range(num_layers):
        if i < num_layers - 1:
            return_sequences = True
        else:
            return_sequences = False

        x = SimpleRNN(neurons_per_layer, return_sequences=return_sequences, name=f'rnn_new_{i}')(x)
        x = Dropout(dropout_rate, name=f'dropout_rnn_new_{i}')(x)

    return x

def add_LSTM_layers(input_layer, num_layers, neurons_per_layer, dropout_rate):
    x = input_layer
    for i in range(num_layers):
        if i < num_layers - 1:
            return_sequences = True
        else:
            return_sequences = False

        x = LSTM(neurons_per_layer, return_sequences=return_sequences, name=f'lstm_new_{i}')(x)
        x = Dropout(dropout_rate, name=f'dropout_lstm_new_{i}')(x)

    return x


def add_GRU_layers(input_layer, num_layers, neurons_per_layer, dropout_rate):
    x = input_layer
    for i in range(num_layers):
        if i < num_layers - 1:
            return_sequences = True
        else:
            return_sequences = False

        x = GRU(neurons_per_layer, return_sequences=return_sequences, name=f'gru_new_{i}')(x)
        x = Dropout(dropout_rate, name=f'dropout_gru_new_{i}')(x)

    return x
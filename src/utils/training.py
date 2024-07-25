import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import utils.preprocessing as pp
import utils.evaluation as eval
import utils.constants as c


def create_simple_rnn_model(input_shape, num_neurons=64, num_layers=2, learning_rate=1e-4, dropout_rate=0.2):
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


def create_lstm_model(input_shape, num_neurons=64, num_layers=2, learning_rate=1e-4, dropout_rate=0.2):
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

def create_gru_model(input_shape, num_neurons=64, num_layers=2, learning_rate=1e-4, dropout_rate=0.2):
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

def get_parameters(param_grid):
    network_type = np.random.choice(param_grid['network_type'])
    window_size = np.random.choice(param_grid['window_size'])
    learning_rate = np.random.uniform(*param_grid['learning_rate'])
    num_layers = np.random.choice(param_grid['num_layers'])
    neurons_per_layer = np.random.choice(param_grid['neurons_per_layer'])
    batch_size = np.random.choice(param_grid['batch_size'])
    
    
    return {
        'network_type': network_type, 
        'window_size': window_size,
        'learning_rate': learning_rate,
        'num_layers': num_layers,
        'neurons_per_layer': neurons_per_layer,
        'batch_size': batch_size
    }


def random_search_rnn(df, param_grid, dataset_info, num_iterations=100):
    best_mse = np.inf
    best_model = None
    best_params = None
    
    metrics = {}
    
    country = dataset_info['country']
    commodity = dataset_info['commodity']
    scaler_path = c.get_scaler_filename(country, commodity)
    
    for _ in range(num_iterations): 
        params = get_parameters(param_grid)       
        
        X_train, y_train, X_val, y_val, X_test, y_test = pp.prepare_data(df, params['window_size'], 0.2, 0.1, scaler_path)
        
        model = create_simple_rnn_model((params['window_size'], 1), params['neurons_per_layer'], params['num_layers'], params['learning_rate'])
        
        if params['network_type'] == 'LSTM':
            model = create_lstm_model((params['window_size'], 1), params['neurons_per_layer'], params['num_layers'], params['learning_rate'])
        elif params['network_type'] == 'GRU':
            model = create_gru_model((params['window_size'], 1), params['neurons_per_layer'], params['num_layers'], params['learning_rate'])
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                
        # Train the model
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=100, batch_size=params['batch_size'], callbacks=[early_stopping], verbose=0)
        
        val_loss = min(history.history['val_loss'])
        val_mae = min(history.history['val_mae'])
        
        y_pred = model.predict(X_test, verbose=0)
        mae, mse = eval.load_scaler_and_evaluate(scaler_path, y_test, y_pred.reshape(-1,))

        # Check if this model has the best validation performance
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_params = {
                'network_type': params['network_type'],
                'window_size': params['window_size'],
                'learning_rate': params['learning_rate'],
                'num_layers': params['num_layers'],
                'neurons_per_layer': params['neurons_per_layer'],
                'batch_size': params['batch_size']
            }
            
            metrics = {
                'val_loss': val_loss,
                'val_mae': val_mae,
                'test_mse': best_mse,
                'test_mae': mae
            }
    
    return best_model, best_params, metrics
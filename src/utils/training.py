import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import utils.preprocessing as pp

def random_search_rnn(df, param_grid, dataset_info, num_iterations=10):
    best_val_loss = np.inf
    best_val_mse = np.inf
    best_model = None
    best_params = None
    
    for _ in range(num_iterations):        
        network_type = np.random.choice(param_grid['network_type'])
        window_size = np.random.choice(param_grid['window_size'])
        learning_rate = np.random.uniform(*param_grid['learning_rate'])
        num_layers = np.random.choice(param_grid['num_layers'])
        neurons_per_layer = np.random.choice(param_grid['neurons_per_layer'])
        batch_size = np.random.choice(param_grid['batch_size'])
        
        # Split the data into training and validation sets
        country = dataset_info['country']
        commodity = dataset_info['commodity']
        X_train, y_train, X_val, y_val = pp.prepare_data(df, window_size, 0.2, f'../scalers/{country}-{commodity}-StandardScaler.pkl')
        
        model =  Sequential()
        
        for i in range(num_layers):
            if network_type == 'RNN':
                model.add(SimpleRNN(neurons_per_layer, activation='relu'))
            elif network_type == 'LSTM':
                model.add(LSTM(neurons_per_layer, activation='relu'))
            elif network_type == 'GRU':
                model.add(GRU(neurons_per_layer, activation='relu'))
            if i < num_layers - 1:
                model.add(Dropout(0.2))
        
        model.add(Dense(1))
        
        # Compile the model
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
        # Train the model
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=100, batch_size=batch_size, callbacks=[early_stopping], verbose=0)
        
        mse = mean_squared_error(y_val, model.predict(X_val))
        
        val_loss = min(history.history['val_loss'])
        
        # Check if this model has the best validation performance
        if val_loss < best_val_loss and mse < best_val_mse:
            best_val_loss = val_loss
            best_val_mse = mse
            best_model = model
            best_params = {
                'network_type': network_type,
                'window_size': window_size,
                'learning_rate': learning_rate,
                'num_layers': num_layers,
                'neurons_per_layer': neurons_per_layer,
                'batch_size': batch_size
            }
    
    return best_model, best_params
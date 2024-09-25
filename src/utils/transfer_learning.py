import tensorflow as tf
import utils.tools as tls
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model, Model
import utils.preprocessing as pp
import utils.evaluation as eval
import utils.constants as c
import numpy as np
import pandas as pd

def add_simple_RNN_layers(input_layer, num_layers, neurons_per_layer):
    x = input_layer
    for i in range(num_layers):
        if i < num_layers - 1:
            return_sequences = True
        else:
            return_sequences = False

        x = SimpleRNN(neurons_per_layer, return_sequences=return_sequences)(x)
        x = Dropout(0.2)(x)

    return x

def add_LSTM_layers(input_layer, num_layers, neurons_per_layer):
    x = input_layer
    for i in range(num_layers):
        if i < num_layers - 1:
            return_sequences = True
        else:
            return_sequences = False

        x = LSTM(neurons_per_layer, return_sequences=return_sequences)(x)
        x = Dropout(0.2)(x)

    return x


def add_GRU_layers(input_layer, num_layers, neurons_per_layer):
    x = input_layer
    for i in range(num_layers):
        if i < num_layers - 1:
            return_sequences = True
        else:
            return_sequences = False

        x = GRU(neurons_per_layer, return_sequences=return_sequences)(x)
        x = Dropout(0.2)(x)

    return x

def prepare_model(pretrained_model, params):
    model_with_removed_output = Model(inputs=pretrained_model.input, 
                                      outputs=pretrained_model.layers[-2].output)

    for layer in model_with_removed_output.layers:
        layer.trainable = False

    reshaped_output = tf.expand_dims(model_with_removed_output.output, axis=1)

    if params['network_type'] == 'RNN':
        x = add_simple_RNN_layers(reshaped_output, params['num_layers'], params['neurons_per_layer'])
    elif params['network_type'] == 'LSTM':
        x = add_LSTM_layers(reshaped_output, params['num_layers'], params['neurons_per_layer'])
    elif params['network_type'] == 'GRU':
        x = add_GRU_layers(reshaped_output, params['num_layers'], params['neurons_per_layer'])
    
    new_output = Dense(1, activation='linear')(x)
    
    new_model = Model(inputs=pretrained_model.input, outputs=new_output)
    return new_model


def transfer_learning(X_train, y_train, X_val, y_val, X_test, y_test, pretrained_model, scaler_path, params):
    new_model = prepare_model(pretrained_model, params)
    
    new_model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                      loss='mean_squared_error', 
                      metrics=['mae'])
    
    print('Compiled')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    new_model.fit(X_train, y_train, 
                  epochs=100, 
                  batch_size=params['batch_size'], 
                  validation_data=(X_val, y_val), 
                  callbacks=[early_stopping], 
                  verbose=0)
    
    y_pred = new_model.predict(X_test, verbose=0)
    mae, mse = eval.load_scaler_and_evaluate(scaler_path, y_test, y_pred.reshape(-1,))
    
    return new_model, mse, mae


def random_search_transfer_learning(df, model_path, window_size, param_grid, scaler_path, num_iterations=100):
    pretrained_model = load_model(model_path)
    
    # Prepare the data
    X_train, y_train, X_val, y_val, X_test, y_test = pp.prepare_data(df, window_size, 0.2, 0.1, scaler_filename=scaler_path)
    
    best_mse = np.inf
    best_model = None
    best_params = None
    metrics = {}
    
    for _ in range(num_iterations): 
        params = tls.get_parameters(param_grid)
        print(params)
        
        model, mse, mae = transfer_learning(X_train, y_train, X_val, y_val, X_test, y_test, pretrained_model, scaler_path, params)
        
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_params = {
                'network_type': params['network_type'],
                'learning_rate': params['learning_rate'],
                'num_layers': params['num_layers'],
                'neurons_per_layer': params['neurons_per_layer'],
                'batch_size': params['batch_size']
            }
            
            metrics = {
                'mse': best_mse,
                'mae': mae
            }
            
    
    return best_model, best_params, metrics
        
        
def transfer_learning_pipeline(target_country, base_countries, commodity, param_grid, json_path):
    for country in base_countries:

        df = pd.read_csv(c.get_countries(commodity, target_country)['processed'])

        base_metadata = tls.get_result(c.get_large_model_results(), country, commodity)
        window_size = base_metadata['best_params']['window_size']

        scaler_path = c.get_scaler_filename(target_country, commodity)

        tl_model, best_params, metrics = random_search_transfer_learning(df[['usdprice']], c.get_model_filename(country, commodity), window_size, param_grid, scaler_path, num_iterations=10)
        print(f"Transfer learning MAE: {metrics['mae']}")
        print(best_params)

        tl_model.save(c.get_tl_model_filename(country, target_country, commodity, 'new-layers'))

        result = {
            'base': country,
            'country': target_country,
            'commodity': commodity,
            'path': c.get_tl_model_filename(country, target_country, commodity, 'new-layers'),
            'best_params': best_params,
            'best_mae': metrics['mae']
        }

        tls.write_tl_results(json_path, result)
        


def transfer_learning_pipeline_small_lr(target_country, base_countries, commodity, json_path, learning_rates):
    for country in base_countries:

        df = pd.read_csv(c.get_countries(commodity, target_country)['processed'])

        base_metadata = tls.get_result(c.get_large_model_results(), country, commodity)
        window_size = base_metadata['best_params']['window_size']
        
        scaler_path = c.get_scaler_filename(target_country, commodity)
        
        X_train, y_train, X_val, y_val, X_test, y_test = pp.prepare_data(df[['usdprice']], window_size, 0.2, 0.1, scaler_filename=scaler_path)
        
        best_model = None
        best_mae = np.inf
        
        for learning_rate in learning_rates:
            pretrained_model = load_model(c.get_model_filename(country, commodity))
            
            pretrained_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                    
            pretrained_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                epochs=100, batch_size=base_metadata['best_params']['batch_size'], callbacks=[early_stopping], verbose=0)
            
            y_pred = pretrained_model.predict(X_test, verbose=0)
            mae, mse = eval.load_scaler_and_evaluate(scaler_path, y_test, y_pred.reshape(-1,))
            
            if mae < best_mae:
                best_mae = mae
                best_model = pretrained_model
        
        print(f"Transfer learning MAE: {mae}")

        best_model.save(c.get_tl_model_filename(country, target_country, commodity, 'small-rate'))

        result = {
            'base': country,
            'country': target_country,
            'commodity': commodity,
            'path': c.get_tl_model_filename(country, target_country, commodity, 'small-rate'),
            'best_mae': best_mae,
            'best_params': base_metadata['best_params']
        }

        tls.write_tl_results(json_path, result)
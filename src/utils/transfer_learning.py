from tensorflow.keras.callbacks import EarlyStopping
import optuna
import pandas as pd
import utils.tools as tls
import utils.preprocessing as pp
import utils.evaluation as eval
import utils.tools as tls
import utils.constants as c
import utils.models as m
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense


def add_layers(trial, reshaped_output):
    network_type = trial.suggest_categorical("network_type", ["RNN", "LSTM", "GRU"])
    num_layers = trial.suggest_int("num_layers", 1, 5)
    num_neurons = trial.suggest_int("num_neurons", 16, 256)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3)
    
    if network_type == 'RNN':
        return m.add_simple_RNN_layers(reshaped_output, num_layers, num_neurons, dropout_rate)
    elif network_type == 'LSTM':
        return m.add_LSTM_layers(reshaped_output, num_layers, num_neurons, dropout_rate)
    elif network_type == 'GRU':
        return m.add_GRU_layers(reshaped_output, num_layers, num_neurons, dropout_rate)


def prepare_model(trial, pretrained_model):
    model_with_removed_output = Model(inputs=pretrained_model.input, 
                                      outputs=pretrained_model.layers[-2].output)

    for layer in model_with_removed_output.layers:
        layer.trainable = False

    reshaped_output = tf.expand_dims(model_with_removed_output.output, axis=1)
    x = add_layers(trial, reshaped_output)

    
    new_output = Dense(1, activation='linear', name='output_new')(x)
    
    new_model = Model(inputs=pretrained_model.input, outputs=new_output)
    return new_model


def objective(trial, X_train, y_train, X_val, y_val, X_test, y_test, pretrained_model, scaler_path):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_int("batch_size", 16, 64)
    
    
    new_model = prepare_model(trial, pretrained_model)
    new_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    print('Compiled')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    new_model.fit(X_train, y_train, 
                  epochs=100, 
                  batch_size=batch_size, 
                  validation_data=(X_val, y_val), 
                  callbacks=[early_stopping], 
                  verbose=0)
    
    y_pred = new_model.predict(X_test, verbose=0)
    mae, _ = eval.load_scaler_and_evaluate(scaler_path, y_test, y_pred.reshape(-1,))
    
    return mae


def bayesian_optimization_params(X_train, y_train, X_val, y_val, X_test, y_test, pretrained_model, scaler_path, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, X_test, y_test, pretrained_model, scaler_path), n_trials=n_trials)
    
    best_params = study.best_params
    print(f"Best parameters: {best_params}")
    
    return best_params



def build_train_evaluate_save_model(X_train, y_train, X_val, y_val, X_test, y_test, pretrained_model, scaler_path, best_params):
    model_with_removed_output = Model(inputs=pretrained_model.input, 
                                      outputs=pretrained_model.layers[-2].output)

    for layer in model_with_removed_output.layers:
        layer.trainable = False

    reshaped_output = tf.expand_dims(model_with_removed_output.output, axis=1)
    
    if best_params['network_type'] == 'RNN':
        x = m.add_simple_RNN_layers(reshaped_output, best_params['num_layers'], best_params['num_neurons'], best_params['dropout_rate'])
    elif best_params['network_type'] == 'LSTM':
        x = m.add_LSTM_layers(reshaped_output, best_params['num_layers'], best_params['num_neurons'], best_params['dropout_rate'])
    elif best_params['network_type'] == 'GRU':
        x = m.add_GRU_layers(reshaped_output, best_params['num_layers'], best_params['num_neurons'], best_params['dropout_rate'])

    
    new_output = Dense(1, activation='linear', name='output_new')(x)
    new_model = Model(inputs=pretrained_model.input, outputs=new_output)
    new_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mse', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    new_model.fit(X_train, y_train, 
                  epochs=100, 
                  batch_size=best_params['batch_size'], 
                  validation_data=(X_val, y_val), 
                  callbacks=[early_stopping], 
                  verbose=0)
    
    y_pred = new_model.predict(X_test, verbose=0)
    mae, _ = eval.load_scaler_and_evaluate(scaler_path, y_test, y_pred.reshape(-1,))
    
    return new_model, mae


def transfer_learning_pipeline(target_country, countries, commodity, json_path, n_trials=100):
    scaler_path = c.get_scaler_filename(target_country, commodity)
    df = pd.read_csv(c.get_countries(commodity, target_country)['processed'])
    
    for country in countries:
        pretrained_model = load_model(c.get_model_filename(country, commodity))
        
        model_metadata = tls.get_result(c.get_large_model_results(), country, commodity)
        window_size = model_metadata['best_params']['window_size']
        
        X_train, y_train, X_val, y_val, X_test, y_test = pp.prepare_data(df[['usdprice']], window_size, 0.2, 0.1, scaler_filename=scaler_path)
        best_params = bayesian_optimization_params(X_train, y_train, X_val, y_val, X_test, y_test, pretrained_model, scaler_path, n_trials)
        model, mae = build_train_evaluate_save_model(X_train, y_train, X_val, y_val, X_test, y_test, pretrained_model, scaler_path, best_params)
        
        model.save(c.get_tl_model_filename(country, target_country, commodity, 'new-layers'))
        
        result = {
            'base': country,
            'country': target_country,
            'commodity': commodity,
            'path': c.get_tl_model_filename(country, target_country, commodity, 'new-layers'),
            'best_params': best_params,
            'best_mae': mae
        }

        tls.write_tl_results(json_path, result)
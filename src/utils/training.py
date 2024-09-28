from tensorflow.keras.callbacks import EarlyStopping
import optuna
import pandas as pd
import utils.tools as tls
import utils.preprocessing as pp
import utils.evaluation as eval
import utils.tools as tls
import utils.constants as c
import utils.models as m


def create_simple_rnn_model(trial, input_shape):
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    num_neurons = trial.suggest_categorical("num_neurons", [16, 32, 64, 128])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 1e-2, 1e-2])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.3])
    
    return m.build_rnn_model(input_shape, num_layers, num_neurons, learning_rate, dropout_rate)

def create_lstm_model(trial, input_shape):
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    num_neurons = trial.suggest_categorical("num_neurons", [16, 32, 64, 128])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 1e-2, 1e-2])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.3])
       
    return m.build_lstm_model(input_shape, num_layers, num_neurons, learning_rate, dropout_rate)

def create_gru_model(trial, input_shape):
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    num_neurons = trial.suggest_categorical("num_neurons", [16, 32, 64, 128])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 1e-2, 1e-2])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.3])
    
    return m.build_gru_model(input_shape, num_layers, num_neurons, learning_rate, dropout_rate)


def build_model(trial, network_type, input_shape):
    if network_type == "RNN":
        return create_simple_rnn_model(trial, input_shape)
    elif network_type == "LSTM":
        return create_lstm_model(trial, input_shape)
    elif network_type == "GRU":
        return create_gru_model(trial, input_shape)


# Define objective function for Optuna
def objective(trial, df, scaler_path):
    window_size = trial.suggest_categorical("window_size", [2, 4, 6])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    network_type = trial.suggest_categorical("network_type", ["RNN", "LSTM", "GRU"])
    
    input_shape = (window_size, 1)    
    X_train, y_train, X_val, y_val, X_test, y_test = pp.prepare_data(df[['usdprice']], window_size, 0.2, 0.1, scaler_filename=scaler_path)
    
    model = build_model(trial, network_type, input_shape)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=100, batch_size=batch_size, callbacks=[early_stopping], verbose=0)
    
    y_pred = model.predict(X_test)
    test_mae, _ = eval.load_scaler_and_evaluate(scaler_path, y_test, y_pred.reshape(-1,))
    
    return test_mae


def bayesian_optimization_params(commodity, country, n_trials):
    study = optuna.create_study(direction="minimize")
    df = pd.read_csv(c.get_countries(commodity, country)['processed'])
    scaler_path = c.get_scaler_filename(country, commodity)
    
    study.optimize(lambda trial: objective(trial, df, scaler_path), n_trials=n_trials)

    best_params = study.best_params
    print(f"Best parameters: {best_params}")
    
    return best_params

def train_save_evaluate_model(commodity, country, model_path, best_params, scaler_path):
    df = pd.read_csv(c.get_countries(commodity, country)['processed'])
    X_train, y_train, X_val, y_val, X_test, y_test = pp.prepare_data(df[['usdprice']], best_params['window_size'], 0.2, 0.1, scaler_filename=scaler_path)
    
    model = m.build_model(best_params, input_shape=(best_params['window_size'], 1))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=100, batch_size=best_params['batch_size'], callbacks=[early_stopping], verbose=0)
    
    model.save(model_path)
    
    y_pred = model.predict(X_test)
    mae, _ = eval.load_scaler_and_evaluate(scaler_path, y_test, y_pred.reshape(-1,))
    
    return mae
    
    
def training_pipeline(countries, commodity, json_path, n_trials=100):
    for country in countries:
        best_params = bayesian_optimization_params(commodity, country, n_trials)
        model_path = c.get_model_filename(country, commodity)
        best_mae = train_save_evaluate_model(commodity, country, model_path, best_params, c.get_scaler_filename(country, commodity))
        
        result = {
            'country': country,
            'commodity': commodity,
            'path': model_path,
            'best_params': best_params,
            'best_mae': best_mae
        }
        
        tls.write_results(json_path, result)
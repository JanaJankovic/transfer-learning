from tensorflow.keras.callbacks import EarlyStopping
import optuna
import pandas as pd
import utils.tools as tls
import utils.preprocessing as pp
import utils.evaluation as eval
import utils.tools as tls
import utils.constants as c
import utils.models as m
import time


def create_simple_rnn_model(trial, input_shape):
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    num_neurons = trial.suggest_categorical("num_neurons", [16, 32, 64, 128])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 1e-2, 1e-2])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.3])

    return m.build_rnn_model(
        input_shape, num_layers, num_neurons, learning_rate, dropout_rate
    )


def create_lstm_model(trial, input_shape):
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    num_neurons = trial.suggest_categorical("num_neurons", [16, 32, 64, 128])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 1e-2, 1e-2])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.3])

    return m.build_lstm_model(
        input_shape, num_layers, num_neurons, learning_rate, dropout_rate
    )


def create_gru_model(trial, input_shape):
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    num_neurons = trial.suggest_categorical("num_neurons", [16, 32, 64, 128])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 1e-2, 1e-2])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.3])

    return m.build_gru_model(
        input_shape, num_layers, num_neurons, learning_rate, dropout_rate
    )


def build_model(trial, network_type, input_shape):
    if network_type == "RNN":
        return create_simple_rnn_model(trial, input_shape)
    elif network_type == "LSTM":
        return create_lstm_model(trial, input_shape)
    elif network_type == "GRU":
        return create_gru_model(trial, input_shape)


# Define objective function for Optuna
def objective(trial, df, scaler_path):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    network_type = trial.suggest_categorical("network_type", ["RNN", "LSTM", "GRU"])

    input_shape = (c.WINDOW_SIZE, 1)
    X_train, y_train, X_val, y_val, X_test, y_test = pp.prepare_data(
        df[["usdprice"]], scaler_filename=scaler_path
    )

    model = build_model(trial, network_type, input_shape)

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0,
    )

    y_pred = model.predict(X_test)
    test_mae, _, _ = eval.load_scaler_and_evaluate(
        scaler_path,
        y_test,
        y_pred.reshape(
            -1,
        ),
    )

    return test_mae


def bayesian_optimization_params(commodity, country, n_trials, market=None):
    study = optuna.create_study(direction="minimize")

    if market is not None:
        df = pd.read_csv(c.get_market_data(market))
        scaler_path = c.get_scaler_filename(market, commodity)
    else:
        df = pd.read_csv(c.get_countries(commodity, country)["processed"])
        scaler_path = c.get_scaler_filename(country, commodity)

    study.optimize(lambda trial: objective(trial, df, scaler_path), n_trials=n_trials)

    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    return best_params


def train_save_evaluate_model(df, model_path, best_params, scaler_path):
    X_train, y_train, X_val, y_val, X_test, y_test = pp.prepare_data(
        df[["usdprice"]], scaler_filename=scaler_path
    )

    model = m.build_model(best_params, input_shape=(c.WINDOW_SIZE, 1))
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=best_params["batch_size"],
        callbacks=[early_stopping],
        verbose=0,
    )

    end_time = time.time()

    model.save(model_path)

    y_pred = model.predict(X_test)
    mae, mse, mape = eval.load_scaler_and_evaluate(
        scaler_path,
        y_test,
        y_pred.reshape(
            -1,
        ),
    )

    elapsed_time = end_time - start_time
    return mae, mse, mape, len(history.epoch), elapsed_time


def final_training(df, best_params, epochs, model_path):
    model = m.build_model(best_params, input_shape=(c.WINDOW_SIZE, 1))

    X_train, y_train = pp.complete_data(df[["usdprice"]])

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=best_params["batch_size"],
        verbose=0,
    )

    model.save(model_path)


def training_pipeline(
    countries, commodity, json_path, final=False, n_trials=500, market=None
):
    for country in countries:
        best_params = bayesian_optimization_params(commodity, country, n_trials, market)

        if market is not None:
            model_path = c.get_model_filename(
                country, commodity, final=False, market=market
            )
            df = pd.read_csv(c.get_market_data(market))
            final_model_path = c.get_model_filename(
                country, commodity, final=True, market=market
            )
            scalar_path = c.get_scaler_filename(market, commodity)

        else:
            model_path = c.get_model_filename(country, commodity, final=False)
            df = pd.read_csv(c.get_countries(commodity, country)["processed"])
            final_model_path = c.get_model_filename(country, commodity, final=True)
            scalar_path = c.get_scaler_filename(country, commodity)

        mae, mse, mape, epochs, elapsed_time = train_save_evaluate_model(
            df, model_path, best_params, scalar_path
        )

        if final:
            final_training(df, best_params, epochs, final_model_path)

        result = {
            "country": country,
            "commodity": commodity,
            "path": model_path,
            "best_params": best_params,
            "evaluation": {
                "mae": mae,
                "mse": mse,
                "mape": mape,
                "epochs": epochs,
                "elapsed_time": elapsed_time,
            },
        }

        if market is not None:
            result["market"] = market

        tls.write_results(json_path, result)

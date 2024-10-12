import pandas as pd
import utils.tools as tls
import utils.preprocessing as pp
import utils.evaluation as eval
import utils.tools as tls
import utils.constants as c
import time

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense


def prepare_model(pretrained_model):
    x = pretrained_model.layers[-2].output

    for layer in pretrained_model.layers:
        layer.trainable = False

    new_output = Dense(1, activation="linear", name="output_new")(x)
    new_model = Model(inputs=pretrained_model.input, outputs=new_output)
    return new_model


def transfer_learning(
    df,
    params,
    pretrained_model,
    scaler_path,
):
    X_train, y_train, X_val, y_val, X_test, y_test = pp.prepare_data(
        df[["usdprice"]], scaler_filename=scaler_path
    )

    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]

    new_model = prepare_model(pretrained_model)
    new_model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    start_time = time.time()

    history = new_model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0,
    )

    end_time = time.time()

    y_pred = new_model.predict(X_test, verbose=0)
    mae, mse, mape = eval.load_scaler_and_evaluate(
        scaler_path,
        y_test,
        y_pred.reshape(
            -1,
        ),
    )

    elapsed_time = end_time - start_time

    return new_model, mae, mse, mape, len(history.epoch), elapsed_time


def transfer_learning_pipeline(target_country, countries, commodity, json_path):
    scaler_path = c.get_scaler_filename(target_country, commodity)
    df = pd.read_csv(c.get_countries(commodity, target_country)["processed"])

    for country in countries:
        pretrained_model = load_model(
            c.get_model_filename(country, commodity, final=True)
        )
        params = tls.get_result(c.get_large_model_results(), country, commodity)

        model, mae, mse, mape, epochs, elapsed_time = transfer_learning(
            df,
            params["best_params"],
            pretrained_model,
            scaler_path,
        )

        model.save(c.get_tl_model_filename(country, target_country, commodity))

        result = {
            "country": country,
            "target_country": target_country,
            "commodity": commodity,
            "path": c.get_tl_model_filename(country, target_country, commodity),
            "evaluation": {
                "mae": mae,
                "mse": mse,
                "mape": mape,
                "epochs": epochs,
                "elapsed_time": elapsed_time,
            },
        }

        tls.write_results(json_path, result)


def transfer_learning_market_pipeline(
    country, commodity, market_target, market_source, json_path
):
    scaler_path = c.get_scaler_filename(market_target, commodity)
    df = pd.read_csv(c.get_market_data(market_target))

    pretrained_model = load_model(
        c.get_model_filename(country, commodity, final=True, market=market_source)
    )
    params = tls.get_result(
        c.get_market_results(), country, commodity, market=market_source
    )
    model, mae, mse, mape, epochs, elapsed_time = transfer_learning(
        df,
        params["best_params"],
        pretrained_model,
        scaler_path,
    )

    model.save(c.get_tl_model_filename(market_source, market_target, commodity))

    result = {
        "country": market_source,
        "target_country": market_target,
        "commodity": commodity,
        "path": c.get_tl_model_filename(market_source, market_target, commodity),
        "evaluation": {
            "mae": mae,
            "mse": mse,
            "mape": mape,
            "epochs": epochs,
            "elapsed_time": elapsed_time,
        },
    }

    tls.write_results(json_path, result)

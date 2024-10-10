import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import utils.preprocessing as pp
import utils.constants as c
import utils.tools as tls
from tensorflow.keras.models import load_model
import pandas as pd


def load_scaler_and_evaluate(scaler_filename, y_test_scaled, y_pred_scaled):
    # Load the scaler
    with open(scaler_filename, "rb") as f:
        scaler = pickle.load(f)

    # Inverse transform the scaled y_test and y_pred
    y_test_original = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Calculate MAE and MSE
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)

    return mae, mse


def evaluate_unscaled(df, scaler_filename, model_path):
    _, _, _, _, X_test, y_test = pp.prepare_data(df[["usdprice"]], scaler_filename)
    model = load_model(model_path)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return mae, mse


def evaluate_countries_unscaled(countries, commodity, json_path):
    mae = []
    mse = []

    for country in countries:
        meta = tls.get_result(json_path, country, commodity)
        model_path = meta["path"]
        scaler_filename = c.get_scaler_filename(country, commodity)
        df = pd.read_csv(c.get_countries(commodity, country)["processed"])
        ae, se = evaluate_unscaled(df, scaler_filename, model_path)

        mae.append(ae)
        mse.append(se)

    return mae, mse

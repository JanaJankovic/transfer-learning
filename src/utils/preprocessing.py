import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from scipy.stats import kstest
import utils.constants as c


def sort_extract_price(data, commodity, criteria=None):
    df = data.copy()

    if criteria is not None:
        df = df[df["commodity"] == criteria]
    else:
        df = df[df["commodity"].str.contains(commodity, case=False, na=False)]

    df = df[df["pricetype"] == "Retail"]
    df["date"] = pd.to_datetime(df["date"])
    df["usdprice"] = pd.to_numeric(df["usdprice"])

    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
    df_avg = df.groupby("date")["usdprice"].mean().round(3).reset_index()

    return df_avg


def sort_extract_price_market(data, commodity, market):
    df = data.copy()

    df = df[df["commodity"].str.contains(commodity, case=False, na=False)]
    df = df[df["market"] == market]

    df = df[df["pricetype"] == "Retail"]
    df["date"] = pd.to_datetime(df["date"])
    df["usdprice"] = pd.to_numeric(df["usdprice"])

    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
    df_avg = df.groupby("date")["usdprice"].mean().round(3).reset_index()

    return df_avg


def split_train_val_test(df, test_size, val_size, window_size, scaler_filename):
    total_data_points = len(df)

    if total_data_points < 30:

        min_data_needed = 3 * (window_size + 1)

        if total_data_points < min_data_needed:
            raise ValueError(
                f"Insufficient data: The dataset must have at least {min_data_needed} instances."
            )

        test_size = window_size + 1
        val_size = window_size + 1
        train_size = total_data_points - (test_size + val_size)

        train_data = df[:train_size]
        val_data = df[train_size : train_size + val_size]
        test_data = df[train_size + val_size :]

    else:
        train_val_data, test_data = train_test_split(
            df, test_size=test_size, shuffle=False
        )
        train_data, val_data = train_test_split(
            train_val_data, test_size=val_size / (1 - test_size), shuffle=False
        )

    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)

    # Save the scaler for later use
    with open(scaler_filename, "wb") as f:
        pickle.dump(scaler, f)

    return train_data_scaled, val_data_scaled, test_data_scaled


def create_sequences(data, window_size):
    X = []
    y = []

    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1]))
    y = y.reshape((y.shape[0],))

    return X, y


def prepare_data(data, scaler_filename):
    window_size = c.WINDOW_SIZE
    test_size = 0.2
    val_size = 0.1

    train_data, val_data_scaled, test_data = split_train_val_test(
        data, test_size, val_size, window_size, scaler_filename
    )

    X_train, y_train = create_sequences(train_data, window_size)
    X_val, y_val = create_sequences(val_data_scaled, window_size)
    X_test, y_test = create_sequences(test_data, window_size)

    return X_train, y_train, X_val, y_val, X_test, y_test


def complete_data(df):
    window_size = c.WINDOW_SIZE
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(df)
    X_train, y_train = create_sequences(train_data, window_size)
    return X_train, y_train


def check_normality(df, country):
    stat, p = kstest(df, "norm")
    text = (
        f"{country}: Statistics={stat:.3f}, p={p:.3f}, Normal"
        if p > 0.05
        else f"{country}: Statistics={stat:.3f}, p={p:.3f}, Not normal"
    )
    print(text)

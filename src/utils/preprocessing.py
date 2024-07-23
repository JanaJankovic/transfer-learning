import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def sort_extract_price(data, commodity):
    df = data.copy()
    
    df = df[df['commodity'].str.contains(commodity, case=False, na=False)]
    df = df[df['pricetype'] == 'Retail']
    
    df['date'] = pd.to_datetime(df['date'])
    df['usdprice'] = pd.to_numeric(df['usdprice'])
    
    df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
    df_avg = df.groupby('date')['usdprice'].mean().round(3).reset_index()
    
    return df_avg

def split_train_test(df, test_size, scaler_filename):
    train_data, test_data = train_test_split(df, test_size=test_size, shuffle=False)

    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    
    return train_data_scaled, test_data_scaled
    
def create_sequences(data, window_size):
    X = []
    y = []
    
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X.shape) == 3 and X.shape[2] == 1:
        X = X.reshape((X.shape[0], X.shape[1]))
    if len(y.shape) == 2 and y.shape[1] == 1:
        y = y.reshape((y.shape[0],))
    
    return X, y

def prepare_data(data, window_size, test_size, scaler_filename):
    train_data, test_data = split_train_test(data, test_size, scaler_filename)
    
    X_train, y_train = create_sequences(train_data, window_size)
    X_test, y_test = create_sequences(test_data, window_size)
    
    return X_train, y_train, X_test, y_test
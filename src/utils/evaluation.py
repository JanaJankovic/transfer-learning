import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_scaler_and_evaluate(scaler_filename, y_test_scaled, y_pred_scaled):
    # Load the scaler
    with open(scaler_filename, 'rb') as f:
        scaler = pickle.load(f)
    
    # Inverse transform the scaled y_test and y_pred
    y_test_original = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate MAE and MSE
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    
    return mae, mse

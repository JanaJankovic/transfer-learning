import matplotlib.pyplot as plt
import utils.preprocessing as pp
import utils.constants as c
import utils.tools as tls
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

def plot_line_charts(dfs, date_column, value_column, titles=None, xlabel='Date', ylabel='Value'):
    num_plots = len(dfs)
    num_rows = (num_plots + 1) // 2 

    fig, axes = plt.subplots(num_rows, 2, figsize=(8, 3 * num_rows))
    axes = axes.flatten()

    for i, df in enumerate(dfs):
        ax = axes[i]
        ax.plot(df[date_column], df[value_column], linestyle='-', color='b')

        ax.set_title(titles[i] if titles else f'Plot {i+1}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)


    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    
def plot_model_prediction(df, model, param_grid, scaler_filename, title):
    _, _, _, _, X_test, _ = pp.prepare_data(df[['usdprice']], param_grid['window_size'], 0.2, 0.1, scaler_filename)
    y_pred = model.predict(X_test)
    
    # Load the scaler
    with open(scaler_filename, 'rb') as f:
        scaler = pickle.load(f)
    
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    dates = pd.to_datetime(df['date'])
    usdprice = df['usdprice']
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(dates, usdprice, label='Actual data', color='blue', linewidth=1)
    
    # Overlay the predictions on the plot
    test_start_index = len(df) - len(y_pred_original)
    
    if len(y_pred_original) == 1:
        # Plot as a single point
        plt.plot(dates.iloc[test_start_index:], y_pred_original, 'ro', label='Predicted data')
    else:
        # Plot as a line
        plt.plot(dates.iloc[test_start_index:], y_pred_original, label='Predicted data', color='red', linewidth=2)
    
    # Adding labels and title
    plt.xlabel('Date')
    plt.ylabel('USD Price')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    
def plot_evaluations(countries, commodity, json_path):
    for country in countries:
        df = pd.read_csv(c.get_countries(commodity, country)['processed'])
        data = tls.get_result(json_path, country, commodity)
        model = load_model(data['path'])
        
        plot_model_prediction(df, model, data['best_params'], c.get_scaler_filename(country, commodity), f'{commodity} price prediction in {country}')
    
    
def plot_tl_evaluations(target_country, countries, commodity, json_path, json_tl_path, type):
    for country in countries:
        df = pd.read_csv(c.get_countries(commodity, target_country)['processed'])
        base_data = tls.get_result(json_path, country, commodity)
        tl_data = tls.get_tl_result(json_tl_path, target_country, country, commodity, c.get_tl_model_filename(country, target_country, commodity, type))
        model = load_model(tl_data['path'])
        
        plot_model_prediction(df, model, base_data['best_params'], c.get_scaler_filename(target_country, commodity), f'{commodity} price prediction in {target_country} learned from {country}')
        
        
def line_bar_plot(df, model, param_grid, scaler_filename):
    _, _, _, _, X_test, _ = pp.prepare_data(df[['usdprice']], param_grid['window_size'], 0.2, 0.1, scaler_filename)
    y_pred = model.predict(X_test)
    
    # Load the scaler
    with open(scaler_filename, 'rb') as f:
        scaler = pickle.load(f)
    
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    dates = pd.to_datetime(df['date'])
    usdprice = df['usdprice']

    return dates, usdprice, y_pred_original



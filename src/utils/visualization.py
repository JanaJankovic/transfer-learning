import matplotlib.pyplot as plt
import utils.preprocessing as pp
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
    
def plot_best_model(df, model, param_grid, scaler_filename, title):
    _, _, _, _, X_test, y_test = pp.prepare_data(df[['usdprice']], param_grid['window_size'], 0.2, 0.1, scaler_filename)
    y_pred = model.predict(X_test)
    
        # Load the scaler
    with open(scaler_filename, 'rb') as f:
        scaler = pickle.load(f)
    
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    dates = pd.to_datetime(df['date'])
    usdprice = df['usdprice']
    
    # Plotting
    plt.figure(figsize=(8, 3))
    plt.plot(dates, usdprice, label='Original Data', color='blue', linewidth=1)
    
    # Overlay the predictions on the plot, aligning them with the correct dates
    test_start_index = len(df) - len(y_pred_original)
    plt.plot(dates.iloc[test_start_index:], y_pred_original, label='Predictions', color='red', linewidth=1)
    
    # Adding labels and title
    plt.xlabel('Date')
    plt.ylabel('USD Price')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    
    
    
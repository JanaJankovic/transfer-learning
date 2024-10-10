from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import utils.preprocessing as pp
import utils.constants as c
import utils.tools as tls
import pickle
import pandas as pd
import numpy as np


def plot_line_charts(
    dfs, date_column, value_column, titles=None, xlabel="Date", ylabel="Value"
):
    num_plots = len(dfs)
    num_rows = (num_plots + 1) // 2

    fig, axes = plt.subplots(num_rows, 2, figsize=(8, 3 * num_rows))
    axes = axes.flatten()

    for i, df in enumerate(dfs):
        ax = axes[i]
        ax.plot(df[date_column], df[value_column], linestyle="-", color="b")

        ax.set_title(titles[i] if titles else f"Plot {i+1}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.tick_params(axis="x", rotation=45)

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_model_prediction(ax, df, model, param_grid, scaler_filename, title):
    _, _, _, _, X_test, _ = pp.prepare_data(df[["usdprice"]], scaler_filename)
    y_pred = model.predict(X_test)

    # Load the scaler
    with open(scaler_filename, "rb") as f:
        scaler = pickle.load(f)

    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    dates = pd.to_datetime(df["date"])
    usdprice = df["usdprice"]

    # Plotting on the provided axis
    ax.plot(dates, usdprice, label="Actual data", color="blue", linewidth=1)

    # Overlay the predictions on the plot
    test_start_index = len(df) - len(y_pred_original)

    if len(y_pred_original) == 1:
        # Plot as a single point
        ax.plot(
            dates.iloc[test_start_index:], y_pred_original, "ro", label="Predicted data"
        )
    else:
        # Plot as a line
        ax.plot(
            dates.iloc[test_start_index:],
            y_pred_original,
            label="Predicted data",
            color="red",
            linewidth=2,
        )

    # Adding labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("USD Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.tick_params(axis="x", rotation=45)


def plot_evaluations(countries, commodity, json_path, figsize=(12, 6)):
    fig, axs = plt.subplots(
        nrows=len(countries) // 2 + len(countries) % 2, ncols=2, figsize=figsize
    )  # Create a grid of subplots with 2 columns
    axs = axs.flatten()  # Flatten the axis array for easier indexing

    for i, country in enumerate(countries):
        df = pd.read_csv(c.get_countries(commodity, country)["processed"])
        data = tls.get_result(json_path, country, commodity)
        model = load_model(data["path"])

        plot_model_prediction(
            axs[i],
            df,
            model,
            data["best_params"],
            c.get_scaler_filename(country, commodity),
            f"{commodity} price prediction in {country}",
        )

    plt.tight_layout()
    plt.show()


def plot_tl_evaluations(
    target_country, countries, commodity, json_path, json_tl_path, type
):
    for country in countries:
        df = pd.read_csv(c.get_countries(commodity, target_country)["processed"])
        base_data = tls.get_result(json_path, country, commodity)
        tl_data = tls.get_tl_result(
            json_tl_path,
            target_country,
            country,
            commodity,
            c.get_tl_model_filename(country, target_country, commodity, type),
        )
        model = load_model(tl_data["path"])

        plot_model_prediction(
            df,
            model,
            base_data["best_params"],
            c.get_scaler_filename(target_country, commodity),
            f"{commodity} price prediction in {target_country} learned from {country}",
        )


def bar_plot_mae(axs, i, target_country, country, commodity):
    bar_values = tls.get_all_metrics(target_country, country, commodity)
    bar_labels = [country, target_country, "Transfer Learning"]

    axs[i, 0].bar(
        range(len(bar_values)), bar_values, color=["#13274f", "#e7a801", "#ce1141"]
    )

    axs[i, 0].set_xticks(range(len(bar_labels)))
    axs[i, 0].set_xticklabels(bar_labels)

    # Set plot title and labels
    axs[i, 0].set_title(f"MAE comparison between models")
    axs[i, 0].set_ylabel("MAE")
    axs[i, 0].set_xlabel("Models")


def line_chart_values(df, scaler_filename, model_path):
    model = load_model(model_path)

    _, _, _, _, X_test, _ = pp.prepare_data(df[["usdprice"]], scaler_filename)
    y_pred = model.predict(X_test)

    with open(scaler_filename, "rb") as f:
        scaler = pickle.load(f)

    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    dates = pd.to_datetime(df["date"])
    usdprice = df["usdprice"]
    return dates, usdprice, y_pred_original


def line_chart_prediction(axs, i, target_country, country, commodity, base=True):
    df = pd.read_csv(c.get_countries(commodity, target_country)["processed"])
    scaler_filename = c.get_scaler_filename(target_country, commodity)

    if base:
        model_path = c.get_model_filename(target_country, commodity)
        title = f"Prediction plot of small model ({target_country})"
        column = 1
    else:
        model_path = c.get_tl_model_filename(
            country, target_country, commodity, "new-layers"
        )
        title = f"Prediction plot of transfer learning model ({country})"
        column = 2

    dates, usdprice, y_pred_original = line_chart_values(
        df, scaler_filename, model_path
    )

    axs[i, column].plot(dates, usdprice, label="Actual data", color="blue", linewidth=1)
    axs[i, column].plot(
        dates.iloc[-len(y_pred_original) :],
        y_pred_original,
        label="Predicted data",
        color="red",
        linewidth=2,
    )
    axs[i, column].set_title(title)
    axs[i, column].set_ylabel("USD Price")
    axs[i, column].legend()
    axs[i, column].grid(True)
    axs[i, column].tick_params(axis="x", rotation=45)


def visualize_tl_summary(target_country, countries, commodity, title):
    num_rows = len(countries)

    fig, axs = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows))

    for i, (country) in enumerate(countries):
        bar_plot_mae(axs, i, target_country, country, commodity)
        line_chart_prediction(axs, i, target_country, country, commodity)
        line_chart_prediction(axs, i, target_country, country, commodity, base=False)

    # Set the overall plot title
    fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to fit the suptitle
    plt.show()


def plot_bar_from_array(
    values,
    labels,
    title="Bar Plot",
    xlabel="X-axis",
    ylabel="Y-axis",
    target_value=None,
):
    """
    Creates a bar plot from an array of values with corresponding labels.

    :param values: List or array of numeric values for the bar heights
    :param labels: List or array of strings for the bar labels
    :param title: Title of the bar plot
    :param xlabel: Label for the x-axis
    :param ylabel: Label for the y-axis
    """

    # Check if the length of values matches the length of labels
    if len(values) != len(labels):
        raise ValueError("The length of 'values' must match the length of 'labels'.")

    # Create the figure and the bar plot
    plt.figure(figsize=(8, 6))  # Adjust the size as needed
    indices = np.arange(len(values))  # The x locations for the bars

    # Create the bar plot
    plt.bar(indices, values, color="skyblue")

    # Set the x-axis ticks to the labels
    plt.xticks(indices, labels)

    # Adding labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    for i, value in enumerate(values):
        plt.text(
            indices[i],
            value + 0.0001,
            f"{round(value, 4)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if target_value is not None:
        plt.axhline(
            y=target_value,
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Target: {target_value}",
        )
        # Add label for the target line
        plt.text(
            len(values) - 0.5,
            target_value + 0.0001,
            f"Base: {target_value}",
            color="red",
            fontsize=10,
        )

    plt.xticks(
        rotation=45, ha="right"
    )  # Rotate labels for better readability if needed
    plt.tight_layout()

    # Show the plot
    plt.show()

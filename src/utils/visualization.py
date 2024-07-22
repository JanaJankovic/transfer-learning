import matplotlib.pyplot as plt

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
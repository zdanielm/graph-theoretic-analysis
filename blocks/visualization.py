import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_distribution(hist_data, value_type):
    """
    Plot a histogram of the given distribution data.

    Parameters
    ----------
    hist_data : dict
        A dictionary containing the distribution data with keys "bin_center" and "probability".
    """

    x = hist_data["bin_center"]
    y = hist_data["probability"]

    plt.figure()
    plt.bar(x, y, width=(x[1] - x[0]))
    plt.title("Distribution")
    plt.xlabel(value_type)
    plt.ylabel("Probability")

    plt.show()


def plot_cumulative_histogram(cumulative_data, value_type):
    """
    Plot a bar chart of the cumulative distribution of the given data.

    Parameters
    ----------
    cumulative_data : dict
        A dictionary containing the cumulative distribution data with keys "upper_limit" and "count".
    """

    x = cumulative_data["upper_limit"]
    y = cumulative_data["count"]

    plt.figure()
    plt.bar(x, y, width=(x[1] - x[0]))
    plt.title("Cumulative distribution")
    plt.xlabel(value_type)
    plt.ylabel("Cumulative count")

    plt.show()


def plot_box(values, value_type):
    """
    Plot a box plot of the given values.

    Parameters
    ----------
    values : list or array_like
        List or array of values to plot

    Returns
    -------
    None

    Notes
    -----
    This function does not return anything, it just plots a box plot of the given values.
    """

    plt.figure()
    plt.boxplot(values)
    plt.title(value_type)

    plt.show()


def plot_temporal_changes(weights: np.ndarray, edges: list, top_edges: list, iterations: list = None):
    """
    Plot the temporal evolution of the top dynamic edges.

    Parameters
    ----------
    - weights: np.ndarray, shape (num_edges, num_iters)
        Weight matrix where each row is an edge, each column is an iteration.
    - edges: list of tuples
        List of edges corresponding to rows in `weights`.
    - top_edges: list of tuples
        List of edges to plot (subset of edges).
    - iterations: list or np.ndarray
        Iteration numbers for x-axis. Defaults to range(num_iters) if None.

    Returns
    -------
    None
    """
    if iterations is None:
        iterations = list(range(weights.shape[1]))

    fig, ax = plt.subplots(figsize=(16, 9))

    top_indices = [edges.index(edge) for edge in top_edges]

    for idx in top_indices:
        ax.plot(
            iterations,
            weights[idx, :],
            marker='o',
            linewidth=3,
            markersize=7,
            label=f"{edges[idx][0]}->{edges[idx][1]}"
        )

    ax.set_xlabel("Iteration", fontsize=18, fontweight=600)
    ax.set_ylabel("Weight", fontsize=18, fontweight=600)
    ax.set_title("Top Dynamic Edge Weights Over Time", fontsize=24, fontweight=800)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_fontweight(600)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        title="Edge ID",
        prop={'size': 16, 'weight': 600},
        title_fontproperties={'size': 16, 'weight': 600}
    )
    ax.grid(True)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()


def QQ_plot(data, dist="norm", network_name=""):
    """
    Creates a Q-Q plot of the given data.

    Parameters
    ----------
    data : array_like
        The data to plot.
    dist : str, optional
        The distribution to compare with, defaults to "norm".
    network_name : str, optional
        The name of the network to include in the plot title, defaults to "".

    Returns
    -------
    None
    """

    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(data, dist=dist, plot=ax)
    ax.set_title(f"Q-Q Plot of Weights ({network_name})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

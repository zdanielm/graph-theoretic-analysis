import matplotlib.pyplot as plt


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
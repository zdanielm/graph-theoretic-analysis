from typing import List

import networkx as nx
import numpy as np
from scipy.stats import anderson, shapiro


def get_attribute(G, attribute="weight"):
    """
    Returns an array of attribute values from all edges in the graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    attribute : str, optional
        Attribute name to fetch, defaults to "weight"

    Returns
    -------
    np.ndarray
        Array of attribute values
    """

    return np.array([
        data.get(attribute, 0)
        for _, _, data in G.edges(data=True)
    ])


def normality_test(graph: nx.DiGraph, attribute: str = "weight"):
    """
    Performs a normality test on the given attribute of the graph.

    If the sample size is less than 5000, the Shapiro-Wilk test is used.
    Otherwise, the Anderson-Darling test is used.

    Parameters
    ----------
    graph : nx.DiGraph
        Input graph
    attribute : str, optional
        Attribute name to fetch, defaults to "weight"

    Returns
    -------
    dict
        A dictionary containing the test name, statistic, p-value, critical values, and significance level.
    """

    samples = get_attribute(graph, attribute)
    if len(samples) < 5000:
        stat, p = shapiro(samples)
        return {
            "test": "shapiro_wilk",
            "statistic": stat,
            "p_value": p,
            "success": bool(p > 0.05)
        }
    else:
        result = anderson(samples, dist="norm")

        sig_levels = np.asarray(result.significance_level)
        crit_vals = np.asarray(result.critical_values)
        idx_5pct = int(np.argmin(np.abs(sig_levels - 5.0)))
        success = bool(result.statistic < crit_vals[idx_5pct])
        return {
            "test": "anderson_darling",
            "statistic": result.statistic,
            "critical_values": result.critical_values,
            "significance_levels": result.significance_level,
            "success": success
        }


def base_stats(samples):
    """
    Calculate basic statistics of a given array of samples.

    Parameters
    ----------
    samples : array_like
        Array of samples

    Returns
    -------
    dict
        A dictionary containing the following statistics:
        - n: number of elements in the array
        - min: minimum value in the array
        - max: maximum value in the array
        - mean: mean value of the array
        - median: median value of the array
        - std: standard deviation of the array (with ddof=1)
    """

    samples = np.asarray(samples)

    return {
        "n": len(samples),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "std": float(np.std(samples, ddof=1)),
    }

def calculate_distribution(samples, bins=100):
    """
    Calculate the histogram of the given samples.

    Parameters
    ----------
    samples : array_like
        Array of samples
    bins : int, optional
        Number of bins in the histogram, defaults to 100

    Returns
    -------
    dict
        A dictionary containing the following information:
        - bin_center: the center of each bin
        - count: the number of elements in each bin
        - probability: the probability of each bin (i.e. the count divided by the total number of samples)
    """

    hist, edges = np.histogram(samples, bins=bins)

    centers = (edges[:-1] + edges[1:]) / 2
    probs = hist / len(samples)

    return {
        "bin_center": centers,
        "count": hist,
        "probability": probs
    }


def cumulative_histogram(samples, bins=100):
    """
    Calculate the cumulative histogram of the given samples.

    Parameters
    ----------
    samples : array_like
        Array of samples
    bins : int, optional
        Number of bins for the histogram (default is 100)

    Returns
    -------
    dict
        A dictionary containing the following statistics:
        - upper_limit: upper limit of each bin
        - count: cumulative count of samples up to each bin
        - probability: cumulative probability of samples up to each bin
    """

    hist, edges = np.histogram(samples, bins=bins)

    cumulative_counts = np.cumsum(hist)
    cumulative_prob = cumulative_counts / len(samples)

    return {
        "upper_limit": edges[1:],
        "c_count": cumulative_counts,
        "c_probability": cumulative_prob
    }


def empirical_cdf(samples):
    """
    Calculate the empirical cumulative distribution function (CDF) of the given samples.

    Parameters
    ----------
    samples : array_like
        Array of samples

    Returns
    -------
    dict
        A dictionary containing the sorted values and the corresponding CDF.
    """

    sorted_w = np.sort(samples)
    n = len(samples)

    return {
        "value": sorted_w,
        "cdf": np.arange(1, n+1) / n
    }


def initialize_weight_matrix(graphs: List[nx.DiGraph]):
    edges = sorted(graphs[0].edges())
    num_edges = len(edges)
    num_iters = len(graphs)

    # Initialize a weight matrix: rows = edges, cols = iterations
    weights = np.zeros((num_edges, num_iters))

    # Fill matrix
    for i, G in enumerate(graphs):
        weights[:, i] = np.array([G[u][v].get('weight', 1.0) for u, v in edges])

    return weights


def temporal_change_analysis(weight_matrix, edges, top_n=5):
    delta_weights = np.diff(weight_matrix, axis=1)  # shape: (num_edges, num_iters-1)

    #rel_change = delta_weights / weight_matrix[:, :-1]

    max_change_per_edge = np.max(np.abs(delta_weights), axis=1)
    top_indices = np.argsort(max_change_per_edge)[-top_n:]  # top N most dynamic edges

    top_edges = [edges[i] for i in top_indices]

    return top_edges


def analyze_changes():
    pass
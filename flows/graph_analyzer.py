from typing import List

import networkx as nx
import pandas as pd
from prefect import flow, get_run_logger, task

from blocks.graph_analytics import (
    base_stats,
    calculate_distribution,
    cumulative_histogram,
    empirical_cdf,
    get_attribute,
    initialize_weight_matrix,
    normality_test,
    temporal_change_analysis,
)

LOGGER = get_run_logger()



@task(name="Distribution Analysis")
def distribution_analysis(Graph: nx.DiGraph, attribute: str = "weight"):
    """
    Calculate the distribution of a given attribute in a graph.

    Parameters
    ----------
    Graph : nx.DiGraph
        Input graph
    attribute : str, optional
        Attribute name to fetch, defaults to "weight"

    Returns
    -------
    dict
        A dictionary containing the following statistics:
        - base_statistics: base statistics of the attribute values (mean, median, mode, std)
        - distribution: histogram of the attribute values
        - cumulative_distribution: cumulative histogram of the attribute values
        - empirical_cdf_result: empirical cumulative distribution function of the attribute values
    """

    attribute_values = get_attribute(Graph, attribute)
    base_statistics = base_stats(attribute_values)
    distribution = calculate_distribution(attribute_values)
    cumulative_distribution = cumulative_histogram(attribute_values)
    empirical_cdf_result = empirical_cdf(attribute_values)

    return base_statistics | distribution | cumulative_distribution | empirical_cdf_result


@task(name="Analyse Single Graph")
def single_graph_analysis(Graph: nx.DiGraph):
    """
    Analyze a single graph.

    Parameters
    ----------
    Graph : nx.DiGraph
        Input graph

    Returns
    -------
    None
    """

    weight_distribution_analysis_result = distribution_analysis(Graph)
    return weight_distribution_analysis_result

@task(name="Temporal Weight Change Analysis")
def temporal_weight_change_analysis(Graphs: List[nx.DiGraph]):
    """
    Analyze the temporal evolution of edge weights in a given graph family.

    Parameters
    ----------
    Graphs : List[nx.DiGraph]
        List of graphs to analyze

    Returns
    -------
    List
        List of top dynamic edges
    """

    weight_matrix = initialize_weight_matrix(Graphs)
    top_edges = temporal_change_analysis(weight_matrix)

    return top_edges


@flow(name="Analyse Graph Family")
def graph_family_analysis(Graphs: List[nx.DiGraph]):
    """
    Analyze a family of graphs.

    Parameters
    ----------
    Graphs : List[nx.DiGraph]
        List of graphs to analyze

    Returns
    -------
    pd.DataFrame
        DataFrame containing the analysis results of the graph family
    """

    LOGGER.info("Weight Distribution Analysis")
    analysis_result_list = []
    for Graph in Graphs:
        name = Graph.graph["name"]
        analysis_result_list.append(single_graph_analysis(Graph))

    LOGGER.info("Temporal Weight Change Analysis")
    top_edges = temporal_weight_change_analysis(Graphs)

    return pd.DataFrame(analysis_result_list)

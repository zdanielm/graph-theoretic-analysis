"""
Graph2Vec algorithm implementation based on:
    Narayanan, Annamalai, et al. Graph2vec: Learning Distributed Representations of Graphs. 2017,
    https://arxiv.org/abs/1707.05005

Extracts a vector representation of a given NetworkX graph, so one can perform similarity search on it.
"""

import networkx as nx
import numpy as np
from collections import defaultdict, Counter
import random
from typing import List, Dict, Tuple, Any

def get_wl_subgraph(n: Any, G: nx.Graph, d: int, cache: Dict[Tuple[Any, int], str] = None) -> str:
    """
    Algorithm 2: GETWLSUBGRAPH (n, G, d)
    Recursively extracts a rooted subgraph representation of depth d around node n.

    Parameters:
    -----------
    n : node identifier
        Root node of the subgraph
    G : nx.Graph
        Input graph with node attribute 'label' (falls back to node ID if missing)
    d : int
        Depth of neighborhood to consider (d >= 0)
    cache : dict, optional
        Memoization cache to avoid redundant computations. Format: {(node, depth): representation}

    Returns:
    --------
    str : String representation of the rooted subgraph of depth d
    """
    if cache is None:
        cache = {}

    key = (n, d)
    if key in cache:
        return cache[key]

    # node label (use node ID as fallback if 'label' attribute missing)
    node_label = G.nodes[n].get('label', str(n))

    if d == 0:
        rep = str(node_label)
    else:
        # representation of current node at depth d-1
        current_rep = get_wl_subgraph(n, G, d - 1, cache)

        # getting sorted representations of neighbors at depth d-1
        neighbor_reps = [
            get_wl_subgraph(neighbor, G, d - 1, cache)
            for neighbor in G.neighbors(n)
        ]
        neighbor_reps.sort()  # Canonical ordering for isomorphism invariance

        # current representation + sorted neighbor representations
        rep = f"{current_rep}[{','.join(neighbor_reps)}]"

    cache[key] = rep
    return rep


def graph2vec(
    graphs: List[nx.Graph],
    D: int,
    delta: int,
    epochs: int,
    alpha: float,
    negative_samples: int = 5,
    seed: int = 42
) -> np.ndarray:
    """
    Algorithm 1: GRAPH2VEC (G, D, δ, ε, α)
    Learns distributed representations of graphs using WL subtree patterns and PV-DBOW.

    Parameters:
    -----------
    graphs : List[nx.Graph]
        List of NetworkX graphs. Each node should have a 'label' attribute (falls back to node ID)
    D : int
        Maximum depth for WL subtree extraction (creates vocabulary of patterns up to depth D)
    delta : int
        Dimensionality of output graph embeddings
    epochs : int
        Number of training epochs
    alpha : float
        Learning rate for SGD updates
    negative_samples : int, optional (default=5)
        Number of negative samples per positive example (for efficient training)
    seed : int, optional (default=42)
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray : Matrix Φ ∈ ℝ^{|G|×δ} of graph embeddings
    """
    random.seed(seed)
    np.random.seed(seed)

    # extracting WL subtree patterns for all graphs (iterative for efficiency)
    print("Extracting WL subtree patterns...")
    all_docs = []  # List of documents (each document = list of patterns for one graph)
    pattern_counter = Counter()

    for G in graphs:
        # initial representations at depth 0
        reps = {node: str(G.nodes[node].get('label', node)) for node in G.nodes()}
        doc = list(reps.values())
        pattern_counter.update(reps.values())

        # iterative WL relabeling for depths 1 to D
        for depth in range(1, D + 1):
            new_reps = {}
            for node in G.nodes():
                # sorted neighbor representations from previous depth
                neighbor_reps = [
                    reps[neighbor] for neighbor in G.neighbors(node)
                ]
                neighbor_reps.sort()

                # representation: current label + sorted neighbor labels
                new_rep = f"{reps[node]}[{','.join(neighbor_reps)}]"
                new_reps[node] = new_rep

            reps = new_reps
            doc.extend(reps.values())
            pattern_counter.update(reps.values())

        all_docs.append(doc)

    # building vocabulary from frequent patterns (min count = 1)
    vocab = [pattern for pattern, _ in pattern_counter.most_common()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size} unique patterns")

    # docuements -> index sequences
    docs_idx = [
        [word2idx[pattern] for pattern in doc if pattern in word2idx]
        for doc in all_docs
    ]

    # initialize embeddings
    num_graphs = len(graphs)
    graph_vectors = np.random.uniform(-0.5/delta, 0.5/delta, (num_graphs, delta))
    word_vectors = np.random.uniform(-0.5/delta, 0.5/delta, (vocab_size, delta))

    # negative sampling distribution (smoothed unigram distribution)
    word_freqs = np.array([pattern_counter[vocab[i]] for i in range(vocab_size)])
    noise_dist = (word_freqs ** 0.75)
    noise_dist /= noise_dist.sum()

    # training loop (PV-DBOW with negative sampling)
    print("Training graph embeddings...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        graph_indices = list(range(num_graphs))
        random.shuffle(graph_indices)

        for g_idx in graph_indices:
            doc = docs_idx[g_idx]
            if not doc:  # empty documents are skipped
                continue

            # sampling patterns from document (stochastic training)
            for _ in range(max(1, len(doc) // 10)):  # Subsample long documents
                target_idx = random.choice(doc)

                # update positive sample
                graph_vec = graph_vectors[g_idx]
                target_vec = word_vectors[target_idx]

                # define negative samples
                neg_indices = np.random.choice(
                    vocab_size,
                    size=negative_samples,
                    p=noise_dist,
                    replace=True
                )

                # Gradient computation
                # positive example: encourage dot product to be high
                prod_pos = np.dot(graph_vec, target_vec)
                grad_pos = -target_vec * (1 - _sigmoid(prod_pos))
                graph_vec += alpha * grad_pos
                word_vectors[target_idx] += alpha * (-graph_vec * (1 - _sigmoid(prod_pos)))
                epoch_loss += -np.log(_sigmoid(prod_pos) + 1e-10)

                # negative examples: encourage dot products to be low
                for neg_idx in neg_indices:
                    neg_vec = word_vectors[neg_idx]
                    prod_neg = np.dot(graph_vec, neg_vec)
                    grad_neg = neg_vec * _sigmoid(prod_neg)
                    graph_vec += alpha * grad_neg
                    word_vectors[neg_idx] += alpha * (graph_vec * _sigmoid(prod_neg))
                    epoch_loss += -np.log(1 - _sigmoid(prod_neg) + 1e-10)

                # graph vetor update
                graph_vectors[g_idx] = graph_vec

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = epoch_loss / (num_graphs * max(1, len(doc) // 10))
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    return graph_vectors


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)



if __name__ == "__main__":
    # dummy graphs
    G1 = nx.karate_club_graph()
    G2 = nx.erdos_renyi_graph(100, 0.1)
    G3 = nx.watts_strogatz_graph(100, 4, 0.1)

    graphs = [G1, G2, G3]

    # GETWLSUBGRAPH
    print("\n=== GETWLSUBGRAPH Examples ===")
    cache = {}
    print("G1 root=1, depth=0:", get_wl_subgraph(1, G1, 0, cache))
    print("G1 root=1, depth=1:", get_wl_subgraph(1, G1, 1, cache))
    print("G1 root=1, depth=2:", get_wl_subgraph(1, G1, 2, cache))

    # GRAPH2VEC
    print("\n=== GRAPH2VEC Training ===")
    embeddings = graph2vec(
        graphs=graphs,
        D=2,          # max WL depth
        delta=384,     # embedding dimension
        epochs=40,    # training epochs
        alpha=0.025,  # learning rate
        seed=42       # random seed for reproducibility
    )

    print("\n=== Learned Embeddings ===")
    for i, emb in enumerate(embeddings):
        print(f"Graph {i} embedding (first 5 dims): {emb[:5]}")

    # verify functions
    print("\n=== Similarity Check ===")
    from scipy.spatial.distance import cosine
    sim_01 = 1 - cosine(embeddings[0], embeddings[1])
    sim_02 = 1 - cosine(embeddings[0], embeddings[2])
    print(f"Similarity(G0, G1): {sim_01:.4f}")
    print(f"Similarity(G0, G2): {sim_02:.4f}")
    print("Note: G0 and G1 are paths (similar structure), G2 is a star (different structure)")
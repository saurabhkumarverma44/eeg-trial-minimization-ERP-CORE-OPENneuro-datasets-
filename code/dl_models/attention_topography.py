# dl_models/attention_topography.py

import numpy as np
import torch


def attention_to_node_topography(edge_index: torch.Tensor, attention: torch.Tensor, n_nodes: int) -> np.ndarray:
    """
    Converts per-edge attention weights (from GATConv's
    return_attention_weights=True) into a per-electrode topography,
    by summing incoming attention at each destination node.

    edge_index: (2, n_edges) -- [source, destination] per edge
    attention: (n_edges, n_heads) or (n_edges, 1) -- attention weight per edge
    n_nodes: number of electrodes in this graph

    Returns: (n_nodes,) topography -- higher value = more attention
    directed toward that electrode, used as this paradigm's GAT-derived
    spatial map for comparison against the published reference topography.
    """
    attn = attention.detach().cpu().numpy()
    if attn.ndim > 1:
        attn = attn.mean(axis=1)  # average across heads if multi-head

    dst = edge_index[1].detach().cpu().numpy()
    topo = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(topo, dst, attn)

    # Normalize by in-degree so nodes aren'''t just favored for having
    # more incoming edges (relevant if the graph isn'''t perfectly
    # symmetric, though our construction is fully-connected/symmetric)
    in_degree = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(in_degree, dst, 1.0)
    in_degree = np.maximum(in_degree, 1.0)
    topo = topo / in_degree

    return topo

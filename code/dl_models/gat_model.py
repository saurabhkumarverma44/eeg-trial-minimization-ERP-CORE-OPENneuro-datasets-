# dl_models/gat_model.py
# Rebuilt to match WBS item 5.2 exactly.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class EEGGAT(nn.Module):
    """
    Graph Attention Network for EEG classification -- WBS 5.2 spec:
    GATConv(5->64, heads=4) -> ELU -> Dropout(0.3) ->
    GATConv(256->32, heads=1) -> GlobalMeanPool -> Linear

    Node features: per-channel relative band power (5-dim), per WBS 5.1.
    Edges: PLV/wPLI weights (edge_dim=4), per WBS 5.1.
    Pooling: GlobalMeanPool, per WBS 5.2 -- kept as specified even
    though an earlier diagnostic on a non-spec architecture found
    concatenation pooling gave higher raw classification accuracy.
    WBS's actual success criterion (item 5.4-5.6) is NFS computed
    from attention weights, not classification accuracy, so we
    build to spec rather than to our own accuracy-optimized variant.
    """

    def __init__(self, in_channels: int = 5, num_classes: int = 2,
                 edge_dim: int = 4, dropout: float = 0.3):
        super().__init__()
        # Layer 1: 5 -> 64, 4 heads -> output dim = 64*4 = 256
        self.gat1 = GATConv(in_channels, 64, heads=4, dropout=dropout, edge_dim=edge_dim)
        # Layer 2: 256 -> 32, 1 head
        self.gat2 = GATConv(64 * 4, 32, heads=1, dropout=dropout, edge_dim=edge_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x, edge_index, edge_attr, batch, return_attention=False):
        """
        x: (total_nodes_in_batch, in_channels)
        edge_index: (2, total_edges_in_batch)
        edge_attr: (total_edges_in_batch, edge_dim)
        batch: (total_nodes_in_batch,)
        return_attention: if True, also returns the layer-2 attention
            weights (needed for WBS 5.4, topography extraction)
        """
        if return_attention:
            x, (edge_index_1, alpha_1) = self.gat1(
                x, edge_index, edge_attr, return_attention_weights=True)
        else:
            x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.dropout(x)

        if return_attention:
            x, (edge_index_2, alpha_2) = self.gat2(
                x, edge_index, edge_attr, return_attention_weights=True)
        else:
            x = self.gat2(x, edge_index, edge_attr)
        x = F.elu(x)

        pooled = global_mean_pool(x, batch)
        out = self.classifier(self.dropout(pooled))

        if return_attention:
            # Return layer-2 attention (closer to the final representation)
            # along with its corresponding edge_index for mapping back
            # to electrode pairs.
            return out, (edge_index_2, alpha_2)
        return out

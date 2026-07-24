import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch

# Constants from notebook that are used in GNNActorCritic
QBER_HARD = 0.11 # Corrected QBER_HARD to 0.11 as per problem description
MAX_SKR = 5_000_000.0

class AttentionMechanism(nn.Module):
    """
    Custom Attention Mechanism for QKD Routing, incorporating:
    - Topological Masking
    - Multi-Head Physics Splitting
    - Hard-Threshold Gating
    - Destination-Aware Queries
    - Resource-Adaptive Weighting
    """
    def __init__(self, query_dim, key_dim, hidden_dim, num_heads,
                 edge_feat_dim, qber_hard_val=QBER_HARD, max_skr_val=MAX_SKR):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.key_projection = nn.Linear(key_dim, hidden_dim)

        # Each head focuses on specific aspects
        self.head_weights = nn.Parameter(torch.rand(num_heads, key_dim)) # For resource-adaptive weighting

        self.final_projection = nn.Linear(hidden_dim, 1)

        self.qber_hard_val = qber_hard_val
        self.max_skr_val = max_skr_val

    def forward(self, current_node_embedding, neighbor_embeddings,
                destination_node_embedding,
                edge_index, edge_attr,
                cur_node_idx, neighbor_node_idxs,
                node_x): # node_x for general node features

        batch_size = current_node_embedding.shape[0]
        num_neighbors = neighbor_embeddings.shape[0]

        # 1. Destination-Aware Queries:
        # Concatenate destination embedding to current node embedding for query generation.
        query_input = torch.cat([current_node_embedding, destination_node_embedding], dim=-1)
        queries = self.query_projection(query_input).view(batch_size, self.num_heads, self.head_dim)

        # 2. Multi-Head Physics Splitting & Resource-Adaptive Weighting:
        # Keys are derived from neighbor embeddings.
        # We can apply adaptive weighting based on resource status here.
        # For simplicity, let's just project neighbor_embeddings for keys for now.
        # More complex adaptive weighting logic can be introduced here based on node_x.
        keys = self.key_projection(neighbor_embeddings).view(num_neighbors, self.num_heads, self.head_dim)

        # Scaled Dot-Product Attention
        scores = torch.einsum('bhd,nhd->bhn', queries, keys) / (self.head_dim ** 0.5)

        # Reshape for final projection
        scores = scores.mean(dim=1) # Average across heads for a single score per neighbor (simplified)

        # Hard-Threshold Gating and Topological Masking - combined with actual QBER/SKR from edge_attr
        # The `GNNActorCritic` will handle filtering of invalid neighbors and hard threshold gating.
        # The responsibility of this attention module is to provide raw attention scores
        # for valid neighbors, which are then masked by GNNActorCritic.
        # So, no direct QBER/SKR masking here, it will be done externally for flexibility.

        return scores.squeeze(0)

class GNNEncoder(nn.Module):
    """
    3-layer message-passing GNN.
    Input:  node features [N, node_dim], edge features [E, edge_dim]
    Output: node embeddings [N, out_dim]
    """
    def __init__(self, node_dim=7, edge_dim=5, hidden=64, out_dim=32,
                 n_layers=3, drop_edge_rate=0.0):
        super().__init__()
        self.drop_edge_rate = drop_edge_rate

        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden), nn.LayerNorm(hidden), nn.GELU())

        self.msg_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden*3, hidden), nn.LayerNorm(hidden), nn.GELU(),
                nn.Linear(hidden, hidden))
            for _ in range(n_layers)
        ])
        self.upd_layers = nn.ModuleList([
            nn.GRUCell(hidden, hidden) for _ in range(n_layers)
        ])
        self.out = nn.Sequential(
            nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim))

    def forward(self, node_x, edge_index, edge_attr):
        if self.training and self.drop_edge_rate > 0:
            num_edges = edge_index.shape[1]
            keep_mask = torch.rand(num_edges, device=edge_index.device) > self.drop_edge_rate
            edge_index = edge_index[:, keep_mask]
            edge_attr  = edge_attr[keep_mask]
            if edge_index.shape[1] == 0:
                # If all edges are dropped, return initial node embeddings
                return self.out(self.node_enc(node_x))

        h   = self.node_enc(node_x)
        e   = self.edge_enc(edge_attr)

        # Check for empty edge_index before proceeding with message passing
        if edge_index.numel() == 0:
            # If no edges, no message passing occurs. Return initial node embeddings.
            return self.out(h)

        src, dst = edge_index[0], edge_index[1]
        N   = h.shape[0]

        for i, (msg_fn, upd_fn) in enumerate(zip(self.msg_layers, self.upd_layers)):
            m   = torch.cat([h[src], h[dst], e], dim=-1)
            m   = msg_fn(m)

            agg = torch.zeros(N, m.shape[-1], device=h.device)
            agg.index_add_(0, dst, m)

            h   = upd_fn(agg, h)

        return self.out(h)


class GNNActorCritic(nn.Module):
    """
    PPO Actor-Critic with shared GNN encoder and custom Attention.
    Actor:  scores next-hop neighbours from current node embedding using Attention
    Critic: global graph pooling -> scalar value
    """
    def __init__(self, node_dim=9, edge_dim=5,
                 hidden=64, gnn_out=32, drop_edge_rate=0.0,
                 num_attention_heads=4):
        super().__init__()
        self.gnn = GNNEncoder(node_dim, edge_dim, hidden, gnn_out,
                              drop_edge_rate=drop_edge_rate)

        # Actor head now uses the custom AttentionMechanism
        self.actor_attention = AttentionMechanism(
            query_dim=gnn_out * 2,  # current_node_emb + destination_node_emb
            key_dim=gnn_out,      # neighbor_embedding
            hidden_dim=hidden,    # for attention projections
            num_heads=num_attention_heads,
            edge_feat_dim=edge_dim # Pass edge_feat_dim here
        )

        self.critic_head = nn.Sequential(
            nn.Linear(gnn_out, 64), nn.GELU(),
            nn.Linear(64, 1))

    def _mask_logits(self, logits, num_neighbors, neighbour_idxs,
                     edge_index, edge_attr, cur_node_idx):
        """Apply feasibility masking: block links with QBER > QBER_HARD or SKR == 0."""
        mask = torch.ones(num_neighbors, dtype=torch.bool, device=logits.device)
        for i, nb_idx in enumerate(neighbour_idxs):
            # Check for forward edge (cur_node_idx -> nb_idx)
            fwd = ((edge_index[0] == cur_node_idx) & (edge_index[1] == nb_idx)).nonzero(as_tuple=True)[0]
            # Check for reverse edge (nb_idx -> cur_node_idx)
            rev = ((edge_index[0] == nb_idx) & (edge_index[1] == cur_node_idx)).nonzero(as_tuple=True)[0]
            match = fwd if fwd.numel() > 0 else rev

            if match.numel() > 0:
                qber_norm = edge_attr[match[0], 2].item()
                skr_norm  = edge_attr[match[0], 3].item()
                if qber_norm > 1.0 or skr_norm <= 0.0:
                    mask[i] = False
            # If no edge found, mark as invalid (cannot route through non-existent link)
            else:
                mask[i] = False
        logits[~mask] = -1e10 # Use a large negative number for masked logits
        return logits

    def forward(self, node_x, edge_index, edge_attr,
                cur_node_idx, neighbour_idxs, dest_node_idx):
        h     = self.gnn(node_x, edge_index, edge_attr)
        value = self.critic_head(h.mean(dim=0, keepdim=True)).squeeze()

        num_neighbors = len(neighbour_idxs)
        if num_neighbors == 0:
            return torch.empty(0, device=node_x.device), value

        current_node_embedding = h[cur_node_idx]
        destination_node_embedding = h[dest_node_idx]
        neighbor_embeddings = h[neighbour_idxs]

        # Use the attention mechanism to compute logits
        logits = self.actor_attention(
            current_node_embedding.unsqueeze(0),
            neighbor_embeddings,
            destination_node_embedding.unsqueeze(0),
            edge_index, edge_attr,
            cur_node_idx, neighbour_idxs,
            node_x # Pass full node features for adaptive weighting
        )

        logits = self._mask_logits(logits, num_neighbors, neighbour_idxs,
                                   edge_index, edge_attr, cur_node_idx)
        return logits, value

    def get_action(self, node_x, edge_index, edge_attr, cur_node_idx,
                   nb_idxs, dest_node_idx, deterministic=False):
        # Initialize log_prob and entropy at the beginning to ensure they are always bound
        log_prob = torch.tensor(0.0, device=node_x.device)
        entropy = torch.tensor(0.0, device=node_x.device)

        logits, value = self.forward(node_x, edge_index, edge_attr,
                                     cur_node_idx, nb_idxs, dest_node_idx)

        if deterministic:
            action = torch.argmax(logits)
            # log_prob and entropy remain 0.0 as initialized
        else:
            # Handle cases where all actions might be masked to -inf
            if torch.isinf(logits).all():
                action = torch.tensor(0, device=logits.device)
                # log_prob and entropy remain 0.0 as initialized
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
        return action.item(), log_prob, entropy, value

    def forward_batch(self, pyg_batch):
        node_x     = pyg_batch.x
        edge_index = pyg_batch.edge_index
        edge_attr  = pyg_batch.edge_attr
        cur_idxs   = pyg_batch.cur_idxs
        nb_idxs    = pyg_batch.nb_idxs
        dest_idxs  = pyg_batch.dest_idxs # Added destination indices to batch

        h = self.gnn(node_x, edge_index, edge_attr)

        # Critic: one value per graph using ptr slicing
        values_b = []
        for g in range(pyg_batch.num_graphs):
            s, e_ = pyg_batch.ptr[g].item(), pyg_batch.ptr[g + 1].item()
            values_b.append(self.critic_head(h[s:e_].mean(dim=0, keepdim=True)).squeeze())
        values_b = torch.stack(values_b)   # (B,)

        # Actor: logits per graph with feasibility masking
        logits_list = []
        for i in range(pyg_batch.num_graphs):
            cur_idx   = cur_idxs[i]
            nb_global = nb_idxs[i]
            dest_idx  = dest_idxs[i]
            n_nb      = len(nb_global)

            if n_nb == 0:
                logits_list.append(torch.empty(0, device=node_x.device))
                continue

            current_node_embedding = h[cur_idx]
            destination_node_embedding = h[dest_idx]
            neighbor_embeddings = h[nb_global]

            logits = self.actor_attention(
                current_node_embedding.unsqueeze(0),
                neighbor_embeddings,
                destination_node_embedding.unsqueeze(0),
                edge_index, edge_attr,
                cur_idx, nb_global,
                pyg_batch.x[pyg_batch.ptr[i].item():pyg_batch.ptr[i+1].item()] # Pass node features for current graph
            )

            logits = self._mask_logits(logits, n_nb, nb_global,
                                       edge_index, edge_attr, cur_idx)
            logits_list.append(logits)

        return logits_list, values_b

#VERSION 3 CODE using GCNConv

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, Linear, SAGEConv, BatchNorm

class CausalHGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        
        # We use GCNConv for the trade flow because it strictly enforces weights
        self.conv1 = HeteroConv({
            ('Sector', 'INPUT_TO', 'Sector'): GCNConv(-1, hidden_channels),
            # Keep SAGE for non-weighted structural links
            ('Sector', 'LOCATED_IN', 'Country'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')
        
        self.bn1 = BatchNorm(hidden_channels)

        self.conv2 = HeteroConv({
            ('Sector', 'INPUT_TO', 'Sector'): GCNConv(-1, hidden_channels),
            ('Sector', 'LOCATED_IN', 'Country'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')
        
        self.bn2 = BatchNorm(hidden_channels)

        self.outcome_head = Linear(hidden_channels, out_channels)
        self.confounder_head = Linear(hidden_channels, hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        # 1. EXTRACT WEIGHTS
        # We must flatten the weights to 1D for GCNConv
        # If weights are missing, GCNConv defaults to 1.0 (The "Blind" bug you found)
        # So we ensure they are passed explicitly.
        
        trade_weights = edge_weight_dict.get(('Sector', 'INPUT_TO', 'Sector'))
        if trade_weights is not None:
             # Ensure shape is 1D (N,) not (N, 1)
             trade_weights = trade_weights.view(-1)

        # 2. LAYER 1
        # We pass the weights specifically to the edge type that needs them
        x_dict = self.conv1(x_dict, edge_index_dict, 
                            edge_weight_dict={('Sector', 'INPUT_TO', 'Sector'): trade_weights})
        
        x_dict = {key: self.bn1(x) for key, x in x_dict.items()}
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # 3. LAYER 2
        x_dict = self.conv2(x_dict, edge_index_dict,
                            edge_weight_dict={('Sector', 'INPUT_TO', 'Sector'): trade_weights})
        
        x_dict = {key: self.bn2(x) for key, x in x_dict.items()}
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        return self.outcome_head(x_dict['Sector']), self.confounder_head(x_dict['Sector'])

#VERSION 3 CODE using HeteroConv BELOW

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import HeteroConv, GraphConv, Linear, GATConv, BatchNorm

# class CausalHGT(torch.nn.Module):
#     def __init__(self, hidden_channels, out_channels, metadata):
#         super().__init__()
        
#         # Layer 1: Message Passing
#         self.conv1 = HeteroConv({
#             ('Sector', 'INPUT_TO', 'Sector'): GraphConv(-1, hidden_channels),
#             ('Sector', 'LOCATED_IN', 'Country'): GraphConv(-1, hidden_channels),
#             ('Event', 'IMPACTS', 'Country'): GATConv((-1, -1), hidden_channels, heads=1, add_self_loops=False),
#         }, aggr='sum')
        
#         # Stability Layer 1
#         self.bn1 = BatchNorm(hidden_channels)

#         # Layer 2: Deeper Propagation
#         self.conv2 = HeteroConv({
#             ('Sector', 'INPUT_TO', 'Sector'): GraphConv(-1, hidden_channels),
#             ('Sector', 'LOCATED_IN', 'Country'): GraphConv(-1, hidden_channels),
#             ('Event', 'IMPACTS', 'Country'): GATConv((-1, -1), hidden_channels, heads=1, add_self_loops=False),
#         }, aggr='sum')
        
#         # Stability Layer 2
#         self.bn2 = BatchNorm(hidden_channels)

#         self.outcome_head = Linear(hidden_channels, out_channels)
#         self.confounder_head = Linear(hidden_channels, hidden_channels)

#     def forward(self, x_dict, edge_index_dict, edge_weight_dict):
#         s2s_weight = edge_weight_dict.get(('Sector', 'INPUT_TO', 'Sector'))
        
#         # --- STEP 0: BACKUP EVENTS ---
#         # Events are "Source Only" nodes. Conv1 will drop them because they don't get updated.
#         # We must save them to put them back in for Layer 2.
#         event_feat = x_dict.get('Event')
        
#         # --- STEP 1: LAYER 1 ---
#         x_dict = self.conv1(x_dict, edge_index_dict, 
#                             edge_weight_dict={('Sector', 'INPUT_TO', 'Sector'): s2s_weight})
        
#         # Normalization (Only applied to updated nodes like Sector/Country)
#         x_dict = {key: self.bn1(x) for key, x in x_dict.items()}
#         x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
#         # --- STEP 2: RESTORE EVENTS ---
#         # Put the backup back into the dictionary so Conv2 can see them
#         if event_feat is not None:
#             x_dict['Event'] = event_feat
        
#         # --- STEP 3: LAYER 2 ---
#         x_dict = self.conv2(x_dict, edge_index_dict,
#                             edge_weight_dict={('Sector', 'INPUT_TO', 'Sector'): s2s_weight})
        
#         # Normalization
#         x_dict = {key: self.bn2(x) for key, x in x_dict.items()}
#         x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
#         return self.outcome_head(x_dict['Sector']), self.confounder_head(x_dict['Sector'])

#VERSION 2 CODE BELOW

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import HeteroConv, SAGEConv, Linear, GATConv

# class CausalHGT(torch.nn.Module):
#     """
#     The Causally-Constrained Heterogeneous Graph Neural Network.
    
#     Architecture:
#     1. HeteroConv Layers: Learn different 'physics' for Trade vs. Politics.
#     2. Dual Heads: One for prediction (GDP), one for causal regularization.
#     """
#     def __init__(self, hidden_channels, out_channels, metadata):
#         super().__init__()
        
#         # 1. HETEROGENEOUS LAYERS
#         self.conv1 = HeteroConv({
#             # Economic Flow: Sector -> Sector
#             ('Sector', 'INPUT_TO', 'Sector'): SAGEConv((-1, -1), hidden_channels),
            
#             # Hierarchy: Sector -> Country
#             ('Sector', 'LOCATED_IN', 'Country'): SAGEConv((-1, -1), hidden_channels),
            
#             # Political Signal: Event -> Country 
#             # add_self_loops=False is CRITICAL for source-target bipartite graphs
#             ('Event', 'IMPACTS', 'Country'): GATConv((-1, -1), hidden_channels, heads=1, add_self_loops=False),
#         }, aggr='sum')

#         self.conv2 = HeteroConv({
#             ('Sector', 'INPUT_TO', 'Sector'): SAGEConv((-1, -1), hidden_channels),
#             ('Sector', 'LOCATED_IN', 'Country'): SAGEConv((-1, -1), hidden_channels),
#             ('Event', 'IMPACTS', 'Country'): GATConv((-1, -1), hidden_channels, heads=1, add_self_loops=False),
#         }, aggr='sum')

#         # 2. OUTPUT HEADS
#         self.outcome_head = Linear(hidden_channels, out_channels)
#         self.confounder_head = Linear(hidden_channels, hidden_channels)

#     def forward(self, x_dict, edge_index_dict):
#         # 0. BACKUP SOURCE NODES
#         # 'Event' nodes have no incoming edges, so conv1 will DROP them from the output.
#         # We must backup the original features to restore them for Layer 2.
#         event_feat = x_dict.get('Event')

#         # Layer 1: Message Passing
#         x_dict = self.conv1(x_dict, edge_index_dict)
#         x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
#         # 1. RESTORE SOURCE NODES
#         # If we had events in the input, put them back into the dict for Layer 2
#         if event_feat is not None:
#             x_dict['Event'] = event_feat

#         # Layer 2: Deeper Ripple Propagation
#         x_dict = self.conv2(x_dict, edge_index_dict)
#         x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
#         # Focus on Sector Embeddings for final prediction
#         sector_emb = x_dict['Sector']
        
#         # Generate Predictions
#         y_pred = self.outcome_head(sector_emb)
#         z_confounder = self.confounder_head(sector_emb)
        
#         return y_pred, z_confounder

#VERSION 1 CODE BELOW

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv

# class TradeSAGE(torch.nn.Module):
#     """
#     Hybrid SCM Component: The Graph Neural Network Encoder.
    
#     It learns how economic shocks propagate through the supply chain structure.
#     Input:  Sector Embeddings (FastRP) + Network Topology (Edges)
#     Output: Predicted Economic Health (Gross Output)
#     """
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(TradeSAGE, self).__init__()
        
#         # Layer 1: Encoder - Aggregates info from immediate trading partners
#         # "Tell me how my suppliers are doing."
#         self.conv1 = SAGEConv(in_channels, hidden_channels)
        
#         # Layer 2: Encoder - Aggregates info from partners of partners
#         # "Tell me how my suppliers' suppliers are doing."
#         self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
#         # Layer 3: Decoder - Translates the learned representation into a real-world value
#         # "Based on all that, predict my Gross Output."
#         self.linear = torch.nn.Linear(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         # x = Node Features (FastRP Vectors)
#         # edge_index = The Trade Network Structure
        
#         # 1. First Hop (Aggregation)
#         x = self.conv1(x, edge_index)
#         x = F.relu(x) # Non-linearity (Economics is rarely linear)
#         x = F.dropout(x, p=0.2, training=self.training) # Regularization

#         # 2. Second Hop (Deep Aggregation)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
        
#         # 3. Decoding / Prediction
#         x = self.linear(x)
#         return x
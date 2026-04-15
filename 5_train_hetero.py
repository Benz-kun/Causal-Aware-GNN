#Version 6

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from db_connection import get_driver, close_driver
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_edge
from model import CausalHGT

# --- CONFIGURATION ---
EPOCHS = 1000           
HIDDEN_DIM = 64
LR = 0.0005
ALPHA_CAUSAL = 0.1
PATIENCE = 90           

def load_weighted_graph(driver):
    print("🚀 Fetching Graph with LINEAR Weights...")
    data = HeteroData()
    
    with driver.session() as session:
        # 1. NODES
        res_sec = session.run("MATCH (s:Sector) RETURN s.uid as id, s.name as name, s.gross_output as y").data()
        df_sec = pd.DataFrame(res_sec)
        sec_map = {uid: i for i, uid in enumerate(df_sec['id'])}
        
        # Use One-Hot Encoding (Generic Type) to force reliance on edges
        sector_types = pd.get_dummies(df_sec['name'])
        data['Sector'].x = torch.tensor(sector_types.values, dtype=torch.float)
        data['Sector'].y = torch.tensor(np.log1p(df_sec['y'].fillna(0).values), dtype=torch.float).view(-1, 1)
        data['Sector'].num_nodes = len(df_sec)

        # Countries
        res_cou = session.run("MATCH (c:Country) RETURN c.iso_code as id").data()
        df_cou = pd.DataFrame(res_cou)
        cou_map = {uid: i for i, uid in enumerate(df_cou['id'])}
        data['Country'].x = torch.ones((len(df_cou), 1), dtype=torch.float)
        data['Country'].num_nodes = len(df_cou)

        # 2. EDGES with LINEAR SCALING
        print("   Loading & Scaling Relationships...")
        res_edges = session.run("MATCH (s1:Sector)-[r:INPUT_TO]->(s2:Sector) RETURN s1.uid as src, s2.uid as tgt, r.weight as w").data()
        
        src = [sec_map[r['src']] for r in res_edges if r['src'] in sec_map and r['tgt'] in sec_map]
        tgt = [sec_map[r['tgt']] for r in res_edges if r['src'] in sec_map and r['tgt'] in sec_map]
        raw_weights = np.array([float(r['w']) for r in res_edges if r['src'] in sec_map and r['tgt'] in sec_map])
        
        # LINEAR MAX SCALING: Preserves the 40% gap accurately
        max_val = raw_weights.max()
        print(f"   (Max Trade: ${max_val:,.0f} -> Scaled to 1.0)")
        scaled_weights = raw_weights / max_val
        
        data['Sector', 'INPUT_TO', 'Sector'].edge_index = torch.tensor([src, tgt], dtype=torch.long)
        data['Sector', 'INPUT_TO', 'Sector'].edge_weight = torch.tensor(scaled_weights, dtype=torch.float)
        
        # Save max_val for validation
        data.max_trade_val = float(max_val)

        # Location Edges
        res_loc = session.run("MATCH (s:Sector)-[:LOCATED_IN]->(c:Country) RETURN s.uid as src, c.iso_code as tgt").data()
        l_src = [sec_map[r['src']] for r in res_loc if r['src'] in sec_map and r['tgt'] in cou_map]
        l_tgt = [cou_map[r['tgt']] for r in res_loc if r['src'] in sec_map and r['tgt'] in cou_map]
        data['Sector', 'LOCATED_IN', 'Country'].edge_index = torch.tensor([l_src, l_tgt], dtype=torch.long)

    return data

def train():
    driver = get_driver()
    if not driver: return
    data = load_weighted_graph(driver)
    close_driver(driver)
    
    model = CausalHGT(hidden_channels=HIDDEN_DIM, out_channels=1, metadata=data.metadata())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # EXPLICIT DICTIONARY CONSTRUCTION
    weight_dict = {
        ('Sector', 'INPUT_TO', 'Sector'): data['Sector', 'INPUT_TO', 'Sector'].edge_weight
    }
    
    print("\n🧠 Training Explicitly Weighted Model...")
    model.train()
    
    for epoch in range(EPOCHS):
        # optimizer.zero_grad()
        # # Pass weights explicitly
        # y_pred, z_conf = model(data.x_dict, data.edge_index_dict, weight_dict)
        # loss_pred = F.mse_loss(y_pred, data['Sector'].y)
        optimizer.zero_grad()
        # --- NEW: DROPEDGE REGULARIZATION (The "Chaos" Tweak) ---
        # Randomly drop 20% of edges (p=0.2) to force robust learning
        edge_index_drop, edge_weight_drop = dropout_edge(
            original_edge_index, 
            edge_attr=original_edge_weight, 
            p=0.2,  # 20% Dropout probability
            training=True
        )
        
        # Create a temporary weight dict for this epoch
        weight_dict_epoch = {
            ('Sector', 'INPUT_TO', 'Sector'): edge_weight_drop
        }
        
        # Update data object temporarily for the model call
        # Note: We pass edge_index_drop to the model manually or update data structure
        # Since HeteroData is complex, the easiest way is to update the edge_index 
        # inside the data object just for this pass (or use the explicit arguments if your model supports it)
        # EASIER IMPLEMENTATION FOR YOU:
        # Just update the dictionary passed to the model!
        # Pass the DROPPED weights and indices
        # NOTE: You need to update your model.forward to accept edge_index as well if you want true dropout
        # But for "Weight Dropout" (setting weights to zero), you can just mask the weights:
        # SIMPLER "WEIGHT DROPOUT" (No complex index slicing needed)
        # Just zero out 20% of the weights randomly
        mask = torch.rand_like(original_edge_weight) > 0.2
        masked_weights = original_edge_weight * mask
        
        weight_dict_epoch = {
            ('Sector', 'INPUT_TO', 'Sector'): masked_weights
        }
        
        # Pass the CHAOTIC weights
        y_pred, z_conf = model(data.x_dict, data.edge_index_dict, weight_dict_epoch)
        # --- END OF TWEAK ---

        loss_pred = F.mse_loss(y_pred, data['Sector'].y)
        loss_causal = torch.mean(torch.abs(z_conf)) 
        total_loss = loss_pred + (ALPHA_CAUSAL * loss_causal)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Prevent Inf
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:03d} | Total: {total_loss.item():.4f}")
            
        if total_loss < 0.001: 
            print(f"   🎯 Converged at Epoch {epoch}")
            break

    # Save model AND scaling factor
    state = {
        'model': model.state_dict(),
        'max_val': data.max_trade_val
    }
    torch.save(state, "models/hetero_model.pth")
    print("🏁 Model Saved.")

if __name__ == "__main__":
    train()

#Version 5 code with flow based training

# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import os
# from db_connection import get_driver, close_driver
# from torch_geometric.data import HeteroData
# from model import CausalHGT

# # --- CONFIGURATION ---
# EPOCHS = 1000           
# HIDDEN_DIM = 64
# LR = 0.001
# ALPHA_CAUSAL = 0.1
# PATIENCE = 15           

# def load_flow_graph(driver):
#     print("🚀 Fetching Graph with Normalized Weights...")
#     data = HeteroData()
    
#     with driver.session() as session:
#         # 1. NODES
#         res_sec = session.run("MATCH (s:Sector) RETURN s.uid as id, s.name as name, s.gross_output as y").data()
#         df_sec = pd.DataFrame(res_sec)
#         sec_map = {uid: i for i, uid in enumerate(df_sec['id'])}
        
#         sector_types = pd.get_dummies(df_sec['name'])
#         data['Sector'].x = torch.tensor(sector_types.values, dtype=torch.float)
#         data['Sector'].y = torch.tensor(np.log1p(df_sec['y'].fillna(0).values), dtype=torch.float).view(-1, 1)
#         data['Sector'].num_nodes = len(df_sec)

#         # Countries
#         res_cou = session.run("MATCH (c:Country) RETURN c.iso_code as id").data()
#         df_cou = pd.DataFrame(res_cou)
#         cou_map = {uid: i for i, uid in enumerate(df_cou['id'])}
#         data['Country'].x = torch.ones((len(df_cou), 1), dtype=torch.float)
#         data['Country'].num_nodes = len(df_cou)
        
#         # Events
#         res_evt = session.run("MATCH (e:Event) RETURN e.url as id").data()
#         if res_evt:
#              df_evt = pd.DataFrame(res_evt)
#              evt_map = {uid: i for i, uid in enumerate(df_evt['id'])}
#              data['Event'].x = torch.ones((len(df_evt), 1), dtype=torch.float)
#              data['Event'].num_nodes = len(df_evt)
#         else:
#              data['Event'].x = torch.zeros((1, 1))
#              data['Event'].num_nodes = 1
#              evt_map = {"DUMMY": 0}

#         # 2. WEIGHTED EDGES
#         print("   Loading & Scaling Relationships...")
#         res_edges = session.run("MATCH (s1:Sector)-[r:INPUT_TO]->(s2:Sector) RETURN s1.uid as src, s2.uid as tgt, r.weight as w").data()
#         src = [sec_map[r['src']] for r in res_edges if r['src'] in sec_map and r['tgt'] in sec_map]
#         tgt = [sec_map[r['tgt']] for r in res_edges if r['src'] in sec_map and r['tgt'] in sec_map]
        
#         # Log-Normalization
#         raw_weights = np.array([float(r['w']) for r in res_edges if r['src'] in sec_map and r['tgt'] in sec_map])
#         scaled_weights = np.log1p(raw_weights)
        
#         data['Sector', 'INPUT_TO', 'Sector'].edge_index = torch.tensor([src, tgt], dtype=torch.long)
#         data['Sector', 'INPUT_TO', 'Sector'].edge_weight = torch.tensor(scaled_weights, dtype=torch.float)

#         # Other Edges
#         res_loc = session.run("MATCH (s:Sector)-[:LOCATED_IN]->(c:Country) RETURN s.uid as src, c.iso_code as tgt").data()
#         l_src = [sec_map[r['src']] for r in res_loc if r['src'] in sec_map and r['tgt'] in cou_map]
#         l_tgt = [cou_map[r['tgt']] for r in res_loc if r['src'] in sec_map and r['tgt'] in cou_map]
#         data['Sector', 'LOCATED_IN', 'Country'].edge_index = torch.tensor([l_src, l_tgt], dtype=torch.long)
        
#         if 'df_evt' in locals():
#              res_evt_edge = session.run("MATCH (e:Event)-[:IMPACTS]->(c:Country) RETURN e.url as src, c.iso_code as tgt").data()
#              e_src = [evt_map[r['src']] for r in res_evt_edge if r['src'] in evt_map and r['tgt'] in cou_map]
#              e_tgt = [cou_map[r['tgt']] for r in res_evt_edge if r['src'] in evt_map and r['tgt'] in cou_map]
#              data['Event', 'IMPACTS', 'Country'].edge_index = torch.tensor([e_src, e_tgt], dtype=torch.long)

#     return data

# def train():
#     driver = get_driver()
#     if not driver: return
#     data = load_flow_graph(driver)
#     close_driver(driver)
    
#     model = CausalHGT(hidden_channels=HIDDEN_DIM, out_channels=1, metadata=data.metadata())
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
#     # --- CRITICAL FIX: MANUALLY BUILD THE DICTIONARY ---
#     # The model was 'blind' before because data.edge_weight_dict doesn't exist automatically.
#     weight_dict = {
#         ('Sector', 'INPUT_TO', 'Sector'): data['Sector', 'INPUT_TO', 'Sector'].edge_weight
#     }
    
#     print("\n🧠 Training Weight-Aware Model (Explicit Mode)...")
#     model.train()
    
#     for epoch in range(EPOCHS):
#         optimizer.zero_grad()
        
#         # Pass the explicit weight_dict
#         y_pred, z_conf = model(data.x_dict, data.edge_index_dict, weight_dict)
        
#         loss_pred = F.mse_loss(y_pred, data['Sector'].y)
#         loss_causal = torch.mean(torch.abs(z_conf)) 
#         total_loss = loss_pred + (ALPHA_CAUSAL * loss_causal)
        
#         total_loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
        
#         if epoch % 10 == 0:
#             print(f"   Epoch {epoch:03d} | Total: {total_loss.item():.4f}")
            
#         if total_loss < 0.001: 
#             print(f"   🎯 Converged at Epoch {epoch}")
#             break

#     os.makedirs("models", exist_ok=True)
#     torch.save(model.state_dict(), "models/hetero_model.pth")
#     print("🏁 Model Saved.")

# if __name__ == "__main__":
#     train()

#Version 4 code

# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import os
# from db_connection import get_driver, close_driver
# from torch_geometric.data import HeteroData
# from torch_geometric.utils import dropout_edge
# from model import CausalHGT

# # --- CONFIGURATION ---
# EPOCHS = 700          
# HIDDEN_DIM = 64
# LR = 0.005
# ALPHA_CAUSAL = 0.1
# PATIENCE = 15           
# EDGE_DROPOUT_RATE = 0.4 # THE EARTHQUAKE: 30% of trade links snap per epoch

# class EarlyStopper:
#     def __init__(self, patience=10):
#         self.patience = patience
#         self.counter = 0
#         self.min_loss = float('inf')

#     def early_stop(self, loss):
#         if loss < self.min_loss:
#             self.min_loss = loss
#             self.counter = 0
#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 return True
#         return False

# def load_hetero_graph(driver):
#     print("🚀 Fetching Graph for Earthquake Training...")
#     data = HeteroData()
    
#     with driver.session() as session:
#         # 1. LOAD NODES
#         print("   Loading 48-dim Embeddings...")
#         query = "MATCH (s:Sector) RETURN s.uid as id, s.embedding as emb, s.gross_output as y"
#         df_sec = pd.DataFrame(session.run(query).data())
        
#         if 'emb' not in df_sec.columns or df_sec['emb'].iloc[0] is None:
#             raise ValueError("❌ Embeddings missing! Run Step 1 first.")
        
#         # Verify Dimension
#         dim = len(df_sec['emb'].iloc[0])
#         print(f"   (Detected Embedding Dimension: {dim})")

#         sec_map = {uid: i for i, uid in enumerate(df_sec['id'])}
#         emb_matrix = np.stack(df_sec['emb'].values)
#         data['Sector'].x = torch.tensor(emb_matrix, dtype=torch.float)
#         data['Sector'].y = torch.tensor(np.log1p(df_sec['y'].fillna(0).values), dtype=torch.float).view(-1, 1)
#         data['Sector'].num_nodes = len(df_sec)

#         # Countries & Events (Dummy)
#         res_cou = session.run("MATCH (c:Country) RETURN c.iso_code as id").data()
#         df_cou = pd.DataFrame(res_cou)
#         cou_map = {uid: i for i, uid in enumerate(df_cou['id'])}
#         data['Country'].x = torch.ones((len(df_cou), 1), dtype=torch.float)
        
#         res_evt = session.run("MATCH (e:Event) RETURN e.url as id").data()
#         if res_evt:
#             df_evt = pd.DataFrame(res_evt)
#             evt_map = {uid: i for i, uid in enumerate(df_evt['id'])}
#             data['Event'].x = torch.ones((len(df_evt), 1), dtype=torch.float)
#         else:
#             data['Event'].x = torch.zeros((1, 1))
#             evt_map = {}

#         # 2. LOAD EDGES
#         print("   Loading Relationships...")
#         def load_edge(cypher, src_m, tgt_m, edge_type):
#             res = session.run(cypher).data()
#             if not res: return
#             src = [src_m[r['src']] for r in res if r['src'] in src_m and r['tgt'] in tgt_m]
#             tgt = [tgt_m[r['tgt']] for r in res if r['src'] in src_m and r['tgt'] in tgt_m]
#             if src:
#                 data[edge_type].edge_index = torch.tensor([src, tgt], dtype=torch.long)

#         load_edge("MATCH (s1:Sector)-[:INPUT_TO]->(s2:Sector) RETURN s1.uid as src, s2.uid as tgt",
#                   sec_map, sec_map, ('Sector', 'INPUT_TO', 'Sector'))
#         load_edge("MATCH (s:Sector)-[:LOCATED_IN]->(c:Country) RETURN s.uid as src, c.iso_code as tgt",
#                   sec_map, cou_map, ('Sector', 'LOCATED_IN', 'Country'))
#         if len(df_evt) > 0:
#             load_edge("MATCH (e:Event)-[:IMPACTS]->(c:Country) RETURN e.url as src, c.iso_code as tgt",
#                       evt_map, cou_map, ('Event', 'IMPACTS', 'Country'))

#     return data

# def train():
#     driver = get_driver()
#     if not driver: return
#     data = load_hetero_graph(driver)
#     close_driver(driver)
    
#     # Initialize Model
#     model = CausalHGT(hidden_channels=HIDDEN_DIM, out_channels=1, metadata=data.metadata())
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     early_stopper = EarlyStopper(patience=PATIENCE)
    
#     print(f"\n🧠 Training with EDGE DROPOUT (Rate: {EDGE_DROPOUT_RATE})...")
#     model.train()
    
#     # Preserve original edges so we can drop from a fresh copy each time
#     original_sector_edges = data['Sector', 'INPUT_TO', 'Sector'].edge_index.clone()
    
#     for epoch in range(EPOCHS):
#         optimizer.zero_grad()
        
#         # --- THE EARTHQUAKE TRICK ---
#         # Randomly remove 30% of supply chain links
#         dropped_edge_index, _ = dropout_edge(
#             original_sector_edges, 
#             p=EDGE_DROPOUT_RATE, 
#             force_undirected=False,
#             training=True
#         )
        
#         # Update the data object temporarily for this forward pass
#         data['Sector', 'INPUT_TO', 'Sector'].edge_index = dropped_edge_index
        
#         # Forward pass on BROKEN graph
#         y_pred, z_confounder = model(data.x_dict, data.edge_index_dict)
        
#         loss_pred = F.mse_loss(y_pred, data['Sector'].y)
#         loss_causal = torch.mean(torch.abs(z_confounder)) 
#         total_loss = loss_pred + (ALPHA_CAUSAL * loss_causal)
        
#         total_loss.backward()
#         optimizer.step()
        
#         if epoch % 10 == 0:
#             print(f"   Epoch {epoch:04d} | Total: {total_loss.item():.4f}")
            
#         if early_stopper.early_stop(total_loss.item()):
#             print(f"🛑 Early Stopping at Epoch {epoch}.")
#             break

#     # Restore original edges before saving? No need, model weights are what matters.
#     torch.save(model.state_dict(), "models/hetero_model.pth")
#     print("🏁 Model Trained on Unstable Grounds & Saved.")

# if __name__ == "__main__":
#     train()

#Version 3 code using blindflod logic

# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import os
# from db_connection import get_driver, close_driver
# from torch_geometric.data import HeteroData
# from model import CausalHGT

# # --- CONFIGURATION ---
# EPOCHS = 1000           
# HIDDEN_DIM = 64
# LR = 0.005
# ALPHA_CAUSAL = 0.1
# PATIENCE = 30           
# MASK_RATE = 0.4  # BLINDFOLD RATE: 40% of sectors will be hidden in each batch

# class EarlyStopper:
#     def __init__(self, patience=10):
#         self.patience = patience
#         self.counter = 0
#         self.min_loss = float('inf')

#     def early_stop(self, loss):
#         if loss < self.min_loss:
#             self.min_loss = loss
#             self.counter = 0
#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 return True
#         return False

# def load_hetero_graph(driver):
#     print("🚀 Fetching Graph for Blindfolded Training...")
#     data = HeteroData()
    
#     with driver.session() as session:
#         # 1. LOAD NODES
#         print("   Loading Node Embeddings...")
#         query = "MATCH (s:Sector) RETURN s.uid as id, s.embedding as emb, s.gross_output as y"
#         df_sec = pd.DataFrame(session.run(query).data())
        
#         if 'emb' not in df_sec.columns or df_sec['emb'].iloc[0] is None:
#             raise ValueError("❌ Embeddings missing! Run src/3_generate_embeddings.py first.")

#         sec_map = {uid: i for i, uid in enumerate(df_sec['id'])}
#         emb_matrix = np.stack(df_sec['emb'].values)
#         data['Sector'].x = torch.tensor(emb_matrix, dtype=torch.float)
#         data['Sector'].y = torch.tensor(np.log1p(df_sec['y'].fillna(0).values), dtype=torch.float).view(-1, 1)
#         data['Sector'].num_nodes = len(df_sec)

#         # Countries
#         res_cou = session.run("MATCH (c:Country) RETURN c.iso_code as id").data()
#         df_cou = pd.DataFrame(res_cou)
#         cou_map = {uid: i for i, uid in enumerate(df_cou['id'])}
#         data['Country'].x = torch.ones((len(df_cou), 1), dtype=torch.float)
#         data['Country'].num_nodes = len(df_cou)
        
#         # Events
#         res_evt = session.run("MATCH (e:Event) RETURN e.url as id").data()
#         if res_evt:
#             df_evt = pd.DataFrame(res_evt)
#             evt_map = {uid: i for i, uid in enumerate(df_evt['id'])}
#             data['Event'].x = torch.ones((len(df_evt), 1), dtype=torch.float)
#             data['Event'].num_nodes = len(df_evt)
#         else:
#             data['Event'].x = torch.zeros((1, 1))
#             data['Event'].num_nodes = 1
#             evt_map = {"DUMMY": 0}

#         # 2. LOAD EDGES
#         print("   Loading Relationships...")
#         def load_edge(cypher, src_m, tgt_m, edge_type):
#             res = session.run(cypher).data()
#             if not res: return
#             src = [src_m[r['src']] for r in res if r['src'] in src_m and r['tgt'] in tgt_m]
#             tgt = [tgt_m[r['tgt']] for r in res if r['src'] in src_m and r['tgt'] in tgt_m]
#             if src:
#                 data[edge_type].edge_index = torch.tensor([src, tgt], dtype=torch.long)

#         load_edge("MATCH (s1:Sector)-[:INPUT_TO]->(s2:Sector) RETURN s1.uid as src, s2.uid as tgt",
#                   sec_map, sec_map, ('Sector', 'INPUT_TO', 'Sector'))
#         load_edge("MATCH (s:Sector)-[:LOCATED_IN]->(c:Country) RETURN s.uid as src, c.iso_code as tgt",
#                   sec_map, cou_map, ('Sector', 'LOCATED_IN', 'Country'))
#         if data['Event'].num_nodes > 0:
#             load_edge("MATCH (e:Event)-[:IMPACTS]->(c:Country) RETURN e.url as src, c.iso_code as tgt",
#                       evt_map, cou_map, ('Event', 'IMPACTS', 'Country'))

#     return data

# def train():
#     driver = get_driver()
#     if not driver: return
#     data = load_hetero_graph(driver)
#     close_driver(driver)
    
#     model = CausalHGT(hidden_channels=HIDDEN_DIM, out_channels=1, metadata=data.metadata())
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     early_stopper = EarlyStopper(patience=PATIENCE)
    
#     print(f"\n🧠 Training with BLINDFOLDING (Mask Rate: {MASK_RATE})...")
#     model.train()
    
#     for epoch in range(EPOCHS):
#         optimizer.zero_grad()
        
#         # --- THE BLINDFOLD TRICK ---
#         # 1. Clone the features so we don't break the original data
#         x_dict_masked = {k: v.clone() for k, v in data.x_dict.items()}
        
#         # 2. Create a random mask for Sectors
#         num_sectors = data['Sector'].num_nodes
#         # Generate random indices to hide
#         mask_indices = torch.randperm(num_sectors)[:int(num_sectors * MASK_RATE)]
        
#         # 3. Set hidden sectors to ZERO (The Blindfold)
#         # This forces the model to use the EDGES to predict these sectors
#         x_dict_masked['Sector'][mask_indices] = 0.0
        
#         # Forward pass with MASKED data
#         y_pred, z_confounder = model(x_dict_masked, data.edge_index_dict)
        
#         loss_pred = F.mse_loss(y_pred, data['Sector'].y)
#         loss_causal = torch.mean(torch.abs(z_confounder)) 
#         total_loss = loss_pred + (ALPHA_CAUSAL * loss_causal)
        
#         total_loss.backward()
#         optimizer.step()
        
#         if epoch % 10 == 0:
#             print(f"   Epoch {epoch:04d} | Total: {total_loss.item():.4f}")
            
#         if early_stopper.early_stop(total_loss.item()):
#             print(f"🛑 Early Stopping at Epoch {epoch}.")
#             break

#     torch.save(model.state_dict(), "models/hetero_model.pth")
#     print("🏁 Model Trained & Saved.")

# if __name__ == "__main__":
#     train()

#Version 2 code using embeddings 

# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import os
# from db_connection import get_driver, close_driver
# from torch_geometric.data import HeteroData
# from model import CausalHGT

# # --- CONFIGURATION ---
# EPOCHS = 1000           
# HIDDEN_DIM = 64
# LR = 0.005
# ALPHA_CAUSAL = 0.1
# PATIENCE = 30           

# class EarlyStopper:
#     def __init__(self, patience=10):
#         self.patience = patience
#         self.counter = 0
#         self.min_loss = float('inf')

#     def early_stop(self, loss):
#         if loss < self.min_loss:
#             self.min_loss = loss
#             self.counter = 0
#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 return True
#         return False

# def load_hetero_graph(driver):
#     print("🚀 Fetching Neuro-Symbolic Graph...")
#     data = HeteroData()
    
#     with driver.session() as session:
#         # 1. LOAD NODES with EMBEDDINGS (The Fix)
#         print("   Loading Node Embeddings...")
        
#         # We fetch 'embedding' (Input) and 'gross_output' (Target)
#         query = """
#         MATCH (s:Sector) 
#         RETURN s.uid as id, s.embedding as emb, s.gross_output as y
#         """
#         df_sec = pd.DataFrame(session.run(query).data())
        
#         # Check if embeddings exist
#         if 'emb' not in df_sec.columns or df_sec['emb'].iloc[0] is None:
#             raise ValueError("❌ Embeddings missing! Run src/3_generate_embeddings.py first.")

#         sec_map = {uid: i for i, uid in enumerate(df_sec['id'])}
        
#         # INPUT: 128-dim Structural Vector (Parsed from list)
#         # We stack the lists into a matrix
#         emb_matrix = np.stack(df_sec['emb'].values)
#         data['Sector'].x = torch.tensor(emb_matrix, dtype=torch.float)
        
#         # TARGET: Log(GDP)
#         data['Sector'].y = torch.tensor(np.log1p(df_sec['y'].fillna(0).values), dtype=torch.float).view(-1, 1)
#         data['Sector'].num_nodes = len(df_sec)

#         # Countries (Dummy Features)
#         res_cou = session.run("MATCH (c:Country) RETURN c.iso_code as id").data()
#         df_cou = pd.DataFrame(res_cou)
#         cou_map = {uid: i for i, uid in enumerate(df_cou['id'])}
#         data['Country'].x = torch.ones((len(df_cou), 1), dtype=torch.float)
#         data['Country'].num_nodes = len(df_cou)
        
#         # Events (Dummy Features)
#         res_evt = session.run("MATCH (e:Event) RETURN e.url as id").data()
#         if res_evt:
#             df_evt = pd.DataFrame(res_evt)
#             evt_map = {uid: i for i, uid in enumerate(df_evt['id'])}
#             data['Event'].x = torch.ones((len(df_evt), 1), dtype=torch.float)
#             data['Event'].num_nodes = len(df_evt)
#         else:
#             print("⚠️ No Events found. Creating dummy.")
#             data['Event'].x = torch.zeros((1, 1))
#             data['Event'].num_nodes = 1
#             evt_map = {"DUMMY": 0}

#         # 2. LOAD EDGES
#         print("   Loading Relationships...")
#         def load_edge(cypher, src_m, tgt_m, edge_type):
#             res = session.run(cypher).data()
#             if not res: return
#             src = [src_m[r['src']] for r in res if r['src'] in src_m and r['tgt'] in tgt_m]
#             tgt = [tgt_m[r['tgt']] for r in res if r['src'] in src_m and r['tgt'] in tgt_m]
#             if src:
#                 data[edge_type].edge_index = torch.tensor([src, tgt], dtype=torch.long)

#         load_edge("MATCH (s1:Sector)-[:INPUT_TO]->(s2:Sector) RETURN s1.uid as src, s2.uid as tgt",
#                   sec_map, sec_map, ('Sector', 'INPUT_TO', 'Sector'))
        
#         load_edge("MATCH (s:Sector)-[:LOCATED_IN]->(c:Country) RETURN s.uid as src, c.iso_code as tgt",
#                   sec_map, cou_map, ('Sector', 'LOCATED_IN', 'Country'))
        
#         if data['Event'].num_nodes > 0:
#             load_edge("MATCH (e:Event)-[:IMPACTS]->(c:Country) RETURN e.url as src, c.iso_code as tgt",
#                       evt_map, cou_map, ('Event', 'IMPACTS', 'Country'))

#     return data

# def train():
#     driver = get_driver()
#     if not driver: return
#     data = load_hetero_graph(driver)
#     close_driver(driver)
    
#     # Initialize Model with NEW Input Dimensions (128 for Sector)
#     model = CausalHGT(hidden_channels=HIDDEN_DIM, out_channels=1, metadata=data.metadata())
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     early_stopper = EarlyStopper(patience=PATIENCE)
    
#     print(f"\n🧠 Training Retrained... (Expect Loss > 0.0)")
#     model.train()
    
#     for epoch in range(EPOCHS):
#         optimizer.zero_grad()
#         y_pred, z_confounder = model(data.x_dict, data.edge_index_dict)
        
#         loss_pred = F.mse_loss(y_pred, data['Sector'].y)
#         loss_causal = torch.mean(torch.abs(z_confounder)) 
#         total_loss = loss_pred + (ALPHA_CAUSAL * loss_causal)
        
#         total_loss.backward()
#         optimizer.step()
        
#         if epoch % 10 == 0:
#             print(f"   Epoch {epoch:04d} | Total: {total_loss.item():.4f} (Pred: {loss_pred.item():.4f})")
            
#         if early_stopper.early_stop(total_loss.item()):
#             print(f"🛑 Early Stopping at Epoch {epoch}.")
#             break

#     torch.save(model.state_dict(), "models/hetero_model.pth")
#     print("🏁 Model Saved.")

# if __name__ == "__main__":
#     train()

#version 1 code using input [CHEATING]

# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import os
# from db_connection import get_driver, close_driver
# from torch_geometric.data import HeteroData
# from model import CausalHGT

# # --- CONFIGURATION ---
# EPOCHS = 1000           # We set this high, but Early Stopping will cut it short
# HIDDEN_DIM = 64
# LR = 0.005
# ALPHA_CAUSAL = 0.1
# PATIENCE = 30           # Stop if no improvement for 30 epochs

# class EarlyStopper:
#     def __init__(self, patience=10, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.min_validation_loss = float('inf')

#     def early_stop(self, validation_loss):
#         if validation_loss < self.min_validation_loss:
#             self.min_validation_loss = validation_loss
#             self.counter = 0
#         elif validation_loss > (self.min_validation_loss + self.min_delta):
#             self.counter += 1
#             if self.counter >= self.patience:
#                 return True
#         return False

# def load_hetero_graph(driver):
#     print("🚀 Fetching Heterogeneous Graph from Neo4j...")
#     data = HeteroData()
    
#     with driver.session() as session:
#         # 1. NODES
#         print("   Loading Node Features...")
        
#         # Sectors
#         res_sec = session.run("MATCH (s:Sector) RETURN s.uid as id, s.gross_output as y").data()
#         df_sec = pd.DataFrame(res_sec)
#         sec_map = {uid: i for i, uid in enumerate(df_sec['id'])}
#         y = torch.tensor(np.log1p(df_sec['y'].fillna(0).values), dtype=torch.float).view(-1, 1)
#         data['Sector'].x = y 
#         data['Sector'].y = y
#         data['Sector'].num_nodes = len(df_sec)

#         # Countries
#         res_cou = session.run("MATCH (c:Country) RETURN c.iso_code as id").data()
#         df_cou = pd.DataFrame(res_cou)
#         cou_map = {uid: i for i, uid in enumerate(df_cou['id'])}
#         data['Country'].x = torch.ones((len(df_cou), 1), dtype=torch.float)
#         data['Country'].num_nodes = len(df_cou)
        
#         # Events (With Safety Check)
#         res_evt = session.run("MATCH (e:Event) RETURN e.url as id").data()
#         if not res_evt:
#              print("⚠️ No Events found. Creating dummy event to prevent crash.")
#              data['Event'].x = torch.zeros((1, 1), dtype=torch.float)
#              data['Event'].num_nodes = 1
#              evt_map = {"DUMMY": 0}
#         else:
#             df_evt = pd.DataFrame(res_evt)
#             evt_map = {uid: i for i, uid in enumerate(df_evt['id'])}
#             data['Event'].x = torch.ones((len(df_evt), 1), dtype=torch.float)
#             data['Event'].num_nodes = len(df_evt)

#         # 2. EDGES
#         print("   Loading Relationships...")
#         def load_edge(cypher, src_map, tgt_map, edge_type):
#             res = session.run(cypher).data()
#             if not res: return
#             src = [src_map[r['src']] for r in res if r['src'] in src_map and r['tgt'] in tgt_map]
#             tgt = [tgt_map[r['tgt']] for r in res if r['src'] in src_map and r['tgt'] in tgt_map]
#             if src:
#                 data[edge_type].edge_index = torch.tensor([src, tgt], dtype=torch.long)

#         load_edge("MATCH (s1:Sector)-[:INPUT_TO]->(s2:Sector) RETURN s1.uid as src, s2.uid as tgt",
#                   sec_map, sec_map, ('Sector', 'INPUT_TO', 'Sector'))
        
#         load_edge("MATCH (s:Sector)-[:LOCATED_IN]->(c:Country) RETURN s.uid as src, c.iso_code as tgt",
#                   sec_map, cou_map, ('Sector', 'LOCATED_IN', 'Country'))
        
#         if data['Event'].num_nodes > 0:
#             load_edge("MATCH (e:Event)-[:IMPACTS]->(c:Country) RETURN e.url as src, c.iso_code as tgt",
#                       evt_map, cou_map, ('Event', 'IMPACTS', 'Country'))

#     return data

# def train():
#     driver = get_driver()
#     if not driver: return
#     data = load_hetero_graph(driver)
#     close_driver(driver)
    
#     if data is None: return

#     # Initialize Model & Early Stopper
#     model = CausalHGT(hidden_channels=HIDDEN_DIM, out_channels=1, metadata=data.metadata())
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     early_stopper = EarlyStopper(patience=PATIENCE)
    
#     print(f"\n🧠 Training Neuro-Symbolic Model (Max Epochs: {EPOCHS})...")
#     model.train()
    
#     for epoch in range(EPOCHS):
#         optimizer.zero_grad()
        
#         # Forward Pass
#         y_pred, z_confounder = model(data.x_dict, data.edge_index_dict)
        
#         # Loss Calculation
#         loss_pred = F.mse_loss(y_pred, data['Sector'].y)
#         loss_causal = torch.mean(torch.abs(z_confounder)) 
#         total_loss = loss_pred + (ALPHA_CAUSAL * loss_causal)
        
#         total_loss.backward()
#         optimizer.step()
        
#         # Logging
#         if epoch % 10 == 0:
#             print(f"   Epoch {epoch:04d} | Total: {total_loss.item():.4f} (Pred: {loss_pred.item():.4f})")
            
#         # Early Stopping Check
#         if early_stopper.early_stop(total_loss.item()):
#             print(f"🛑 Early Stopping triggered at Epoch {epoch} (No improvement for {PATIENCE} epochs).")
#             break

#     # Save
#     os.makedirs("models", exist_ok=True)
#     save_path = os.path.join("models", "hetero_model.pth")
#     torch.save(model.state_dict(), save_path)
#     print(f"🏁 Training Complete. Model saved to: {save_path}")

# if __name__ == "__main__":
#     train()
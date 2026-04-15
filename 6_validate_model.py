#Test 5

import torch
import pandas as pd
import numpy as np
from db_connection import get_driver, close_driver
from torch_geometric.data import HeteroData
from model import CausalHGT

def load_inference_system():
    # 1. LOAD CHECKPOINT
    # We use 'model' and 'max_val' because that is what src/5_train_hetero.py saved.
    try:
        checkpoint = torch.load("models/hetero_model.pth")
    except FileNotFoundError:
        print("❌ Error: models/hetero_model.pth not found. Please run src/5_train_hetero.py first.")
        return None, None, None

    max_val = checkpoint['max_val']
    print(f"   Loading Scaling Factor: Max Trade = ${max_val:,.0f}")

    driver = get_driver()
    with driver.session() as session:
        # 2. NODES
        res_sec = session.run("MATCH (s:Sector) RETURN s.uid as id, s.name as name, s.country as country, s.gross_output as y").data()
        df_sec = pd.DataFrame(res_sec)
        sec_map = {uid: i for i, uid in enumerate(df_sec['id'])}
        
        x_sec = torch.tensor(pd.get_dummies(df_sec['name']).values, dtype=torch.float)
        y_sec = torch.tensor(np.log1p(df_sec['y'].fillna(0).values), dtype=torch.float).view(-1, 1)

        # 3. EDGES
        res_edges = session.run("MATCH (s1:Sector)-[r:INPUT_TO]->(s2:Sector) RETURN s1.uid as src, s2.uid as tgt, r.weight as w").data()
        src = [sec_map[r['src']] for r in res_edges]
        tgt = [sec_map[r['tgt']] for r in res_edges]
        
        # Apply Saved Scaling
        raw_weights = np.array([float(r['w']) for r in res_edges])
        scaled_weights = raw_weights / max_val
        
        data = HeteroData()
        data['Sector'].x = x_sec
        data['Sector'].y = y_sec
        data['Sector', 'INPUT_TO', 'Sector'].edge_index = torch.tensor([src, tgt], dtype=torch.long)
        data['Sector', 'INPUT_TO', 'Sector'].edge_weight = torch.tensor(scaled_weights, dtype=torch.float)
        
        # Countries (Dummy)
        res_cou = session.run("MATCH (c:Country) RETURN c.iso_code as id").data()
        df_cou = pd.DataFrame(res_cou)
        cou_map = {uid: i for i, uid in enumerate(df_cou['id'])}
        data['Country'].x = torch.ones((len(df_cou), 1))
        
        res_loc = session.run("MATCH (s:Sector)-[:LOCATED_IN]->(c:Country) RETURN s.uid as src, c.iso_code as tgt").data()
        l_src = [sec_map[r['src']] for r in res_loc]
        l_tgt = [cou_map[r['tgt']] for r in res_loc]
        data['Sector', 'LOCATED_IN', 'Country'].edge_index = torch.tensor([l_src, l_tgt], dtype=torch.long)

    close_driver(driver)
    
    # 4. LOAD MODEL
    model = CausalHGT(64, 1, data.metadata())
    # FIX: Using the correct key 'model' instead of 'model_state'
    model.load_state_dict(checkpoint['model']) 
    model.eval()
    return model, data, df_sec

def run_structural_test():
    print("🧪 STARTING 40% LINEAR SANCTION VALIDATION...")
    model, data, df_sec = load_inference_system()
    if not model: return
    
    # 1. BASELINE
    weight_dict_base = {('Sector', 'INPUT_TO', 'Sector'): data['Sector', 'INPUT_TO', 'Sector'].edge_weight}
    with torch.no_grad():
        y_base, _ = model(data.x_dict, data.edge_index_dict, weight_dict_base)
        val_base = torch.expm1(y_base).numpy().flatten()

    # 2. INTERVENTION
    target_uid = "CHN_C26"
    if target_uid not in df_sec['id'].values:
        print(f"❌ Error: {target_uid} not found in database.")
        return

    target_idx = df_sec[df_sec['id'] == target_uid].index[0]
    
    edge_index = data['Sector', 'INPUT_TO', 'Sector'].edge_index
    edge_weight = data['Sector', 'INPUT_TO', 'Sector'].edge_weight.clone()
    
    mask = (edge_index[0] == target_idx) | (edge_index[1] == target_idx)
    print(f"   Targeting {target_uid}: Reducing trade volume by 40% on {mask.sum()} edges.")
    
    # 3. APPLY 40% CUT
    edge_weight[mask] = edge_weight[mask] * 0.6 
    
    # 4. PREDICT SHOCK
    weight_dict_shock = {('Sector', 'INPUT_TO', 'Sector'): edge_weight}
    with torch.no_grad():
        y_shock, _ = model(data.x_dict, data.edge_index_dict, weight_dict_shock)
        val_shock = torch.expm1(y_shock).numpy().flatten()

    # 5. ANALYSIS
    diff = val_base - val_shock
    df_sec['loss'] = diff
    
    # Filter for non-China victims with noticeable loss
    ripples = df_sec[(df_sec['country'] != 'CHN') & (df_sec['loss'] > 0.01)].sort_values('loss', ascending=False)

    print(f"\n[RESULTS] Top Victims of 40% Sanction:")
    if not ripples.empty:
        print(ripples[['id', 'loss']].head(10))
        print(f"\n✅ SUCCESS: Ripples Detected.")
    else:
        print("❌ FAILURE: No impact.")

if __name__ == "__main__":
    run_structural_test()

#TEST 4

# import torch
# import pandas as pd
# import numpy as np
# from db_connection import get_driver, close_driver
# from torch_geometric.data import HeteroData
# from model import CausalHGT

# def load_inference_system():
#     driver = get_driver()
#     if not driver: return None, None, None

#     with driver.session() as session:
#         # Nodes
#         res_sec = session.run("MATCH (s:Sector) RETURN s.uid as id, s.name as name, s.country as country, s.gross_output as y").data()
#         df_sec = pd.DataFrame(res_sec)
#         sec_map = {uid: i for i, uid in enumerate(df_sec['id'])}
        
#         x_sec = torch.tensor(pd.get_dummies(df_sec['name']).values, dtype=torch.float)
#         y_sec = torch.tensor(np.log1p(df_sec['y'].fillna(0).values), dtype=torch.float).view(-1, 1)

#         # Weighted Edges
#         res_edges = session.run("MATCH (s1:Sector)-[r:INPUT_TO]->(s2:Sector) RETURN s1.uid as src, s2.uid as tgt, r.weight as w").data()
#         src = [sec_map[r['src']] for r in res_edges]
#         tgt = [sec_map[r['tgt']] for r in res_edges]
        
#         # Log-Normalize
#         raw_weights = np.array([float(r['w']) for r in res_edges])
#         scaled_weights = np.log1p(raw_weights)
        
#         data = HeteroData()
#         data['Sector'].x = x_sec
#         data['Sector'].y = y_sec
#         data['Sector', 'INPUT_TO', 'Sector'].edge_index = torch.tensor([src, tgt], dtype=torch.long)
#         data['Sector', 'INPUT_TO', 'Sector'].edge_weight = torch.tensor(scaled_weights, dtype=torch.float)

#         # Countries
#         res_cou = session.run("MATCH (c:Country) RETURN c.iso_code as id").data()
#         df_cou = pd.DataFrame(res_cou)
#         cou_map = {uid: i for i, uid in enumerate(df_cou['id'])}
#         data['Country'].x = torch.ones((len(df_cou), 1))
        
#         res_loc = session.run("MATCH (s:Sector)-[:LOCATED_IN]->(c:Country) RETURN s.uid as src, c.iso_code as tgt").data()
#         l_src = [sec_map[r['src']] for r in res_loc]
#         l_tgt = [cou_map[r['tgt']] for r in res_loc]
#         data['Sector', 'LOCATED_IN', 'Country'].edge_index = torch.tensor([l_src, l_tgt], dtype=torch.long)

#     close_driver(driver)
    
#     model = CausalHGT(64, 1, data.metadata())
#     model.load_state_dict(torch.load("models/hetero_model.pth"))
#     model.eval()
#     return model, data, df_sec

# def run_structural_test():
#     print("🧪 STARTING 40% SANCTION VALIDATION...")
#     model, data, df_sec = load_inference_system()
    
#     # 1. BASELINE
#     # Construct dict manually
#     weight_dict_base = {
#         ('Sector', 'INPUT_TO', 'Sector'): data['Sector', 'INPUT_TO', 'Sector'].edge_weight
#     }
#     with torch.no_grad():
#         y_base, _ = model(data.x_dict, data.edge_index_dict, weight_dict_base)
#         val_base = torch.expm1(y_base).numpy().flatten()

#     # 2. INTERVENTION
#     target_uid = "CHN_C26"
#     target_idx = df_sec[df_sec['id'] == target_uid].index[0]
    
#     edge_index = data['Sector', 'INPUT_TO', 'Sector'].edge_index
#     edge_weight = data['Sector', 'INPUT_TO', 'Sector'].edge_weight.clone()
    
#     mask = (edge_index[0] == target_idx) | (edge_index[1] == target_idx)
#     print(f"   Targeting {target_uid}: Reducing trade volume by 40% on {mask.sum()} edges.")
    
#     # Apply 40% Reduction
#     edge_weight[mask] = edge_weight[mask] * 0.6 
    
#     # 3. PREDICT SHOCK
#     # Pass the MODIFIED weights explicitly
#     weight_dict_shock = {
#         ('Sector', 'INPUT_TO', 'Sector'): edge_weight
#     }

#     with torch.no_grad():
#         y_shock, _ = model(data.x_dict, data.edge_index_dict, weight_dict_shock)
#         val_shock = torch.expm1(y_shock).numpy().flatten()

#     # 4. ANALYSIS
#     diff = val_base - val_shock
#     df_sec['loss'] = diff
    
#     ripples = df_sec[(df_sec['country'] != 'CHN') & (df_sec['loss'] > 1.0)].sort_values('loss', ascending=False)

#     print(f"\n[RESULTS] Top Victims of 40% Sanction:")
#     if not ripples.empty:
#         print(ripples[['id', 'loss']].head(10))
#         print(f"\n✅ SUCCESS: Realistic ripples detected.")
#     else:
#         print("❌ FAILURE: No impact detected.")

# if __name__ == "__main__":
#     run_structural_test()

#Test 3

# import torch
# import pandas as pd
# import numpy as np
# import os
# from db_connection import get_driver, close_driver
# from torch_geometric.data import HeteroData
# from model import CausalHGT

# # --- CONFIGURATION ---
# MODEL_PATH = "models/hetero_model.pth"
# HIDDEN_DIM = 64

# def load_system_for_inference():
#     driver = get_driver()
#     if not driver: return None, None, None

#     with driver.session() as session:
#         # 1. Fetch Node Features (MUST MATCH TRAINING)
#         # We now fetch the EMBEDDING, not just the GDP
#         print("   Loading Neuro-Symbolic Features...")
#         query = """
#         MATCH (s:Sector) 
#         RETURN s.uid as id, s.country as country, s.embedding as emb, s.gross_output as y
#         """
#         df_sec = pd.DataFrame(session.run(query).data())
        
#         # Validation Check
#         if 'emb' not in df_sec.columns or df_sec['emb'].iloc[0] is None:
#             print("❌ Error: Embeddings not found. Did you run src/3_generate_embeddings.py?")
#             return None, None, None

#         # Mappings
#         sec_map = {uid: i for i, uid in enumerate(df_sec['id'])}
        
#         # Build Data Object
#         data = HeteroData()
        
#         # INPUT: The 128-dim Embedding Vector
#         emb_matrix = np.stack(df_sec['emb'].values)
#         data['Sector'].x = torch.tensor(emb_matrix, dtype=torch.float)
        
#         # Load other nodes (Dummy)
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

#         # Edges
#         def get_edges(cypher, src_m, tgt_m):
#             r = session.run(cypher).data()
#             if not r: return torch.empty((2, 0), dtype=torch.long)
#             src = [src_m[x['src']] for x in r if x['src'] in src_m and x['tgt'] in tgt_m]
#             tgt = [tgt_m[x['tgt']] for x in r if x['src'] in src_m and x['tgt'] in tgt_m]
#             return torch.tensor([src, tgt], dtype=torch.long)

#         data['Sector', 'INPUT_TO', 'Sector'].edge_index = get_edges(
#             "MATCH (s1:Sector)-[:INPUT_TO]->(s2:Sector) RETURN s1.uid as src, s2.uid as tgt", sec_map, sec_map)
#         data['Sector', 'LOCATED_IN', 'Country'].edge_index = get_edges(
#             "MATCH (s:Sector)-[:LOCATED_IN]->(c:Country) RETURN s.uid as src, c.iso_code as tgt", sec_map, cou_map)
#         if len(df_evt) > 0:
#             data['Event', 'IMPACTS', 'Country'].edge_index = get_edges(
#                 "MATCH (e:Event)-[:IMPACTS]->(c:Country) RETURN e.url as src, c.iso_code as tgt", evt_map, cou_map)
        
#     close_driver(driver)
    
#     # Load Model
#     try:
#         model = CausalHGT(hidden_channels=HIDDEN_DIM, out_channels=1, metadata=data.metadata())
#         model.load_state_dict(torch.load(MODEL_PATH))
#         model.eval()
#     except Exception as e:
#         print(f"❌ Model Load Error: {e}")
#         return None, None, None
    
#     return model, data, df_sec

# def run_tests():
#     print("🧪 STARTING RIPPLE VALIDATION...")
#     model, data, df_sec = load_system_for_inference()
#     if not model: return

#     # 1. BASELINE
#     with torch.no_grad():
#         pred_base, _ = model(data.x_dict, data.edge_index_dict)
#         base_gdp = torch.expm1(pred_base).numpy().flatten()
    
#     # --- GLOBAL RIPPLE TEST ---
#     print("\n[TEST] Shocking CHINA Electronics (CHN_C26)...")
#     target_sector = "CHN_C26"
    
#     if target_sector in df_sec['id'].values:
#         # Apply Shock: Zero out the embedding
#         target_idx = df_sec[df_sec['id'] == target_sector].index
#         x_shock = data.x_dict.copy()
#         x_shock['Sector'] = x_shock['Sector'].clone()
        
#         # NUCLEAR OPTION: We delete the embedding information entirely
#         x_shock['Sector'][target_idx] = torch.zeros(48) 
        
#         with torch.no_grad():
#             pred_shock, _ = model(x_shock, data.edge_index_dict)
#             diff = base_gdp - torch.expm1(pred_shock).numpy().flatten()
            
#         df_sec['loss'] = diff
        
#         # Show Top Victims OUTSIDE China
#         # We look for ANY loss > $1M
#         top_victims = df_sec[
#             (df_sec['country'] != 'CHN') & 
#             (df_sec['loss'] > 1.0)
#         ].sort_values('loss', ascending=False).head(10)
        
#         print("\n   Top International Victims (Ripple Effect):")
#         if not top_victims.empty:
#             print(top_victims[['id', 'loss']])
#             print("\n   ✅ SUCCESS: The Graph is Connected. Ripples Detected.")
#         else:
#             print(top_victims)
#             print("\n   ❌ FAILURE: No Ripples. The model is ignoring the edges.")
#             print("      (It simply memorized that China=Rich and ignored the shock).")

# if __name__ == "__main__":
#     run_tests()

#TEST 2

# import torch
# import pandas as pd
# import numpy as np
# import os
# from db_connection import get_driver, close_driver
# from torch_geometric.data import HeteroData
# from model import CausalHGT

# # --- CONFIGURATION ---
# MODEL_PATH = "models/hetero_model.pth"
# HIDDEN_DIM = 64

# def load_system_for_inference():
#     driver = get_driver()
#     if not driver: return None, None, None

#     with driver.session() as session:
#         # Load Raw Data
#         df_sec = pd.DataFrame(session.run("MATCH (s:Sector) RETURN s.uid as id, s.country as country, s.gross_output as y").data())
#         df_cou = pd.DataFrame(session.run("MATCH (c:Country) RETURN c.iso_code as id").data())
#         df_evt = pd.DataFrame(session.run("MATCH (e:Event) RETURN e.url as id").data())
        
#         # Mappings
#         sec_map = {uid: i for i, uid in enumerate(df_sec['id'])}
#         cou_map = {uid: i for i, uid in enumerate(df_cou['id'])}
#         evt_map = {uid: i for i, uid in enumerate(df_evt['id'])}

#         # Build Data
#         data = HeteroData()
#         data['Sector'].x = torch.tensor(np.log1p(df_sec['y'].fillna(0).values), dtype=torch.float).view(-1, 1)
#         data['Country'].x = torch.ones((len(df_cou), 1), dtype=torch.float)
#         # Handle case where events might be 0
#         if len(df_evt) > 0:
#             data['Event'].x = torch.ones((len(df_evt), 1), dtype=torch.float)
#         else:
#             data['Event'].x = torch.zeros((1, 1)) # Dummy

#         # Edges
#         def get_edges(cypher, src_m, tgt_m):
#             r = session.run(cypher).data()
#             if not r: return torch.empty((2, 0), dtype=torch.long)
#             src = [src_m[x['src']] for x in r if x['src'] in src_m and x['tgt'] in tgt_m]
#             tgt = [tgt_m[x['tgt']] for x in r if x['src'] in src_m and x['tgt'] in tgt_m]
#             return torch.tensor([src, tgt], dtype=torch.long)

#         data['Sector', 'INPUT_TO', 'Sector'].edge_index = get_edges(
#             "MATCH (s1:Sector)-[:INPUT_TO]->(s2:Sector) RETURN s1.uid as src, s2.uid as tgt", sec_map, sec_map)
#         data['Sector', 'LOCATED_IN', 'Country'].edge_index = get_edges(
#             "MATCH (s:Sector)-[:LOCATED_IN]->(c:Country) RETURN s.uid as src, c.iso_code as tgt", sec_map, cou_map)
#         if len(df_evt) > 0:
#             data['Event', 'IMPACTS', 'Country'].edge_index = get_edges(
#                 "MATCH (e:Event)-[:IMPACTS]->(c:Country) RETURN e.url as src, c.iso_code as tgt", evt_map, cou_map)
        
#     close_driver(driver)
    
#     # Load Model
#     try:
#         model = CausalHGT(hidden_channels=HIDDEN_DIM, out_channels=1, metadata=data.metadata())
#         model.load_state_dict(torch.load(MODEL_PATH))
#         model.eval()
#     except Exception as e:
#         print(f"❌ Model Load Error: {e}")
#         return None, None, None
    
#     return model, data, df_sec

# def run_tests():
#     print("🧪 STARTING SCIENTIFIC VALIDATION (GLOBAL CHECK)...")
#     model, data, df_sec = load_system_for_inference()
#     if not model: return

#     # 1. BASELINE
#     with torch.no_grad():
#         pred_base, _ = model(data.x_dict, data.edge_index_dict)
#         base_gdp = torch.expm1(pred_base).numpy().flatten()
    
#     # --- GLOBAL CAPABILITY TEST ---
#     # We will look for Electronics (C26) in China to prove it works outside USA
#     print("\n[TEST B] Ripple Test: Shocking CHINA Electronics (CHN_C26)...")
    
#     # Corrected Search: Looking for 'C26' (WIOD Code) instead of 'Electronics'
#     target_sector = "CHN_C26"
    
#     if target_sector in df_sec['id'].values:
#         print(f"   ✅ Found Sector: {target_sector}")
        
#         # Apply Shock
#         target_idx = df_sec[df_sec['id'] == target_sector].index
#         x_shock = data.x_dict.copy()
#         x_shock['Sector'] = x_shock['Sector'].clone()
#         x_shock['Sector'][target_idx] *= 0.0 # Total Sanction
        
#         with torch.no_grad():
#             pred_shock, _ = model(x_shock, data.edge_index_dict)
#             diff = base_gdp - torch.expm1(pred_shock).numpy().flatten()
            
#         df_sec['loss'] = diff
        
#         # Show Top Victims OUTSIDE China
#         top_victims = df_sec[df_sec['country'] != 'CHN'].sort_values('loss', ascending=False).head(5)
        
#         print("\n   Top International Victims of a China Shock:")
#         print(top_victims[['id', 'loss']])
        
#         if top_victims['loss'].sum() > 0:
#             print("\n   ✅ SUCCESS: Ripples detected across borders.")
#         else:
#             print("\n   ⚠️ WARNING: Shock contained. Check edge weights.")
            
#     else:
#         print(f"   ❌ Error: Could not find {target_sector}. Available IDs look like: {df_sec['id'].iloc[0]}")

# if __name__ == "__main__":
#     run_tests()

#TEST 1 

# import torch
# import pandas as pd
# import numpy as np
# import os
# from db_connection import get_driver, close_driver
# from torch_geometric.data import HeteroData
# from model import CausalHGT

# # --- CONFIGURATION ---
# MODEL_PATH = "models/hetero_model.pth"
# HIDDEN_DIM = 64

# def load_system_for_inference():
#     driver = get_driver()
#     if not driver: return None, None

#     # REBUILD GRAPH (Must match training exactly)
#     with driver.session() as session:
#         # Load Maps
#         df_sec = pd.DataFrame(session.run("MATCH (s:Sector) RETURN s.uid as id, s.country as country, s.gross_output as y").data())
#         df_cou = pd.DataFrame(session.run("MATCH (c:Country) RETURN c.iso_code as id").data())
#         df_evt = pd.DataFrame(session.run("MATCH (e:Event) RETURN e.url as id").data())
        
#         sec_map = {uid: i for i, uid in enumerate(df_sec['id'])}
#         cou_map = {uid: i for i, uid in enumerate(df_cou['id'])}
#         evt_map = {uid: i for i, uid in enumerate(df_evt['id'])}

#         # Build Data
#         data = HeteroData()
#         data['Sector'].x = torch.tensor(np.log1p(df_sec['y'].fillna(0).values), dtype=torch.float).view(-1, 1)
#         data['Country'].x = torch.ones((len(df_cou), 1), dtype=torch.float)
#         data['Event'].x = torch.ones((len(df_evt), 1), dtype=torch.float)
        
#         # Edges
#         def get_edges(cypher, src_m, tgt_m):
#             r = session.run(cypher).data()
#             src = [src_m[x['src']] for x in r if x['src'] in src_m and x['tgt'] in tgt_m]
#             tgt = [tgt_m[x['tgt']] for x in r if x['src'] in src_m and x['tgt'] in tgt_m]
#             return torch.tensor([src, tgt], dtype=torch.long)

#         data['Sector', 'INPUT_TO', 'Sector'].edge_index = get_edges(
#             "MATCH (s1:Sector)-[:INPUT_TO]->(s2:Sector) RETURN s1.uid as src, s2.uid as tgt", sec_map, sec_map)
#         data['Sector', 'LOCATED_IN', 'Country'].edge_index = get_edges(
#             "MATCH (s:Sector)-[:LOCATED_IN]->(c:Country) RETURN s.uid as src, c.iso_code as tgt", sec_map, cou_map)
#         data['Event', 'IMPACTS', 'Country'].edge_index = get_edges(
#             "MATCH (e:Event)-[:IMPACTS]->(c:Country) RETURN e.url as src, c.iso_code as tgt", evt_map, cou_map)
        
#     close_driver(driver)
    
#     # Load Model
#     model = CausalHGT(hidden_channels=HIDDEN_DIM, out_channels=1, metadata=data.metadata())
#     model.load_state_dict(torch.load(MODEL_PATH))
#     model.eval()
    
#     return model, data, df_sec

# def run_tests():
#     print("🧪 STARTING SCIENTIFIC VALIDATION...")
#     model, data, df_sec = load_system_for_inference()
#     if not model: return

#     # 1. BASELINE
#     with torch.no_grad():
#         pred_base, _ = model(data.x_dict, data.edge_index_dict)
#         base_gdp = torch.expm1(pred_base).numpy().flatten()
    
#     # --- TEST A: THE PLACEBO (Sanity Check) ---
#     # We shock a tiny country (e.g., 'CYP' Cyprus or 'MLT' Malta if present, else a small node)
#     # If the model crashes or predicts global collapse, it is OVERTRAINED.
#     print("\n[TEST A] Placebo Test: Shocking a minor player...")
    
#     # Find a small country
#     candidates = ['CYP', 'MLT', 'EST', 'LUX'] 
#     small_country = next((c for c in candidates if c in df_sec['country'].values), None)
    
#     if small_country:
#         print(f"   Selected Placebo Target: {small_country}")
#         x_shock = data.x_dict.copy()
#         x_shock['Sector'] = x_shock['Sector'].clone()
        
#         target_indices = df_sec[df_sec['country'] == small_country].index
#         x_shock['Sector'][target_indices] *= 0.5 # 50% shock
        
#         with torch.no_grad():
#             pred_shock, _ = model(x_shock, data.edge_index_dict)
#             diff = base_gdp - torch.expm1(pred_shock).numpy().flatten()
            
#         global_loss = diff.sum()
#         print(f"   Global Impact: ${global_loss:,.2f} M")
#         if global_loss > 100000: # Arbitrary large number
#             print("   ❌ FAILED: Model is hallucinating global collapse from a tiny shock.")
#         else:
#             print("   ✅ PASSED: Impact is localized and minimal.")
#     else:
#         print("   ⚠️ Skipped: No small country found in dataset.")

#     # --- TEST B: THE STRUCTURAL RIPPLE (Causality Check) ---
#     # Shock USA Electronics. Does it hit MEX Automotive? (Known supply chain link)
#     print("\n[TEST B] Ripple Test: Shocking USA Electronics...")
    
#     usa_elec = df_sec[(df_sec['country'] == 'USA') & (df_sec['id'].str.contains('Electronics|Computer'))].index
#     if not usa_elec.empty:
#         x_shock = data.x_dict.copy()
#         x_shock['Sector'] = x_shock['Sector'].clone()
#         x_shock['Sector'][usa_elec] *= 0.0 # COMPLETE ANNIHILATION
        
#         with torch.no_grad():
#             pred_shock, _ = model(x_shock, data.edge_index_dict)
#             diff = base_gdp - torch.expm1(pred_shock).numpy().flatten()
            
#         df_sec['loss'] = diff
#         top_victims = df_sec[df_sec['country'] != 'USA'].sort_values('loss', ascending=False).head(5)
        
#         print("   Top International Victims:")
#         print(top_victims[['id', 'loss']])
        
#         if top_victims['loss'].sum() > 0:
#             print("   ✅ PASSED: Shocks are propagating across borders.")
#         else:
#             print("   ❌ FAILED: Shock stayed inside USA (Model is unconnected).")
#     else:
#         print("   ⚠️ Skipped: USA Electronics sector not found.")

# if __name__ == "__main__":
#     run_tests()
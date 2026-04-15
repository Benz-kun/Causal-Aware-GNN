#Version 2

import torch
import pandas as pd
import numpy as np
from db_connection import get_driver, close_driver
from torch_geometric.data import HeteroData
from model import CausalHGT

def load_inference_system():
    try:
        checkpoint = torch.load("models/hetero_model.pth")
        max_val = checkpoint['max_val']
    except:
        return None, None, None

    driver = get_driver()
    with driver.session() as session:
        # Nodes
        res_sec = session.run("MATCH (s:Sector) RETURN s.uid as id, s.name as name, s.country as country, s.gross_output as y").data()
        df_sec = pd.DataFrame(res_sec)
        sec_map = {uid: i for i, uid in enumerate(df_sec['id'])}
        
        x_sec = torch.tensor(pd.get_dummies(df_sec['name']).values, dtype=torch.float)
        y_sec = torch.tensor(np.log1p(df_sec['y'].fillna(0).values), dtype=torch.float).view(-1, 1)

        # Edges
        res_edges = session.run("MATCH (s1:Sector)-[r:INPUT_TO]->(s2:Sector) RETURN s1.uid as src, s2.uid as tgt, r.weight as w").data()
        src = [sec_map[r['src']] for r in res_edges]
        tgt = [sec_map[r['tgt']] for r in res_edges]
        
        # Linear Scale
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
    
    model = CausalHGT(64, 1, data.metadata())
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, data, df_sec

def simulate_shock(target_uid, model, data, df_sec):
    print(f"\n[SIMULATION] Targeting {target_uid}...")
    
    # Baseline
    weight_dict_base = {('Sector', 'INPUT_TO', 'Sector'): data['Sector', 'INPUT_TO', 'Sector'].edge_weight}
    with torch.no_grad():
        y_base, _ = model(data.x_dict, data.edge_index_dict, weight_dict_base)
        val_base = torch.expm1(y_base).numpy().flatten()

    # Intervention
    if target_uid not in df_sec['id'].values:
        print(f"   ❌ {target_uid} not found.")
        return None

    target_idx = df_sec[df_sec['id'] == target_uid].index[0]
    edge_index = data['Sector', 'INPUT_TO', 'Sector'].edge_index
    edge_weight = data['Sector', 'INPUT_TO', 'Sector'].edge_weight.clone()
    
    # 40% Blockade
    mask = (edge_index[0] == target_idx) | (edge_index[1] == target_idx)
    edge_weight[mask] = edge_weight[mask] * 0.6 
    
    # Predict
    weight_dict_shock = {('Sector', 'INPUT_TO', 'Sector'): edge_weight}
    with torch.no_grad():
        y_shock, _ = model(data.x_dict, data.edge_index_dict, weight_dict_shock)
        val_shock = torch.expm1(y_shock).numpy().flatten()

    # Calculate Loss
    df_res = df_sec.copy()
    df_res['loss'] = val_base - val_shock
    
    # --- FILTER REMOVED: SHOW RAW TOP 10 REGARDLESS OF MAGNITUDE ---
    target_country = target_uid.split('_')[0]
    ripples = df_res[df_res['country'] != target_country].sort_values('loss', ascending=False)
    
    return ripples[['id', 'loss']]

def run_comparison():
    print("🧪 STARTING GEOPOLITICAL SCENARIO COMPARISON (UNFILTERED)...")
    model, data, df_sec = load_inference_system()
    if not model: return

    # Scenario A: China Electronics Shock
    res_china = simulate_shock("CHN_C26", model, data, df_sec)
    
    # Scenario B: Taiwan Electronics Shock
    res_taiwan = simulate_shock("TWN_C26", model, data, df_sec)
    
    print("\n=======================================================")
    print(" 🇨🇳 SCENARIO A: CHINA SHOCK VICTIMS (Top 5)")
    print("=======================================================")
    if res_china is not None: print(res_china.head(5))
    
    print("\n=======================================================")
    print(" 🇹🇼 SCENARIO B: TAIWAN SHOCK VICTIMS (Top 5)")
    print("=======================================================")
    if res_taiwan is not None: print(res_taiwan.head(5))
    
    print("\n✅ Comparison Complete.")

if __name__ == "__main__":
    run_comparison()

#Version 1

# import torch
# import pandas as pd
# import numpy as np
# from db_connection import get_driver, close_driver
# from torch_geometric.data import HeteroData
# from model import CausalHGT

# def load_inference_system():
#     # Load scaling factor
#     try:
#         checkpoint = torch.load("models/hetero_model.pth")
#         max_val = checkpoint['max_val']
#     except:
#         print("❌ Model not found. Run training first.")
#         return None, None, None

#     driver = get_driver()
#     with driver.session() as session:
#         # Nodes
#         res_sec = session.run("MATCH (s:Sector) RETURN s.uid as id, s.name as name, s.country as country, s.gross_output as y").data()
#         df_sec = pd.DataFrame(res_sec)
#         sec_map = {uid: i for i, uid in enumerate(df_sec['id'])}
        
#         x_sec = torch.tensor(pd.get_dummies(df_sec['name']).values, dtype=torch.float)
#         y_sec = torch.tensor(np.log1p(df_sec['y'].fillna(0).values), dtype=torch.float).view(-1, 1)

#         # Edges
#         res_edges = session.run("MATCH (s1:Sector)-[r:INPUT_TO]->(s2:Sector) RETURN s1.uid as src, s2.uid as tgt, r.weight as w").data()
#         src = [sec_map[r['src']] for r in res_edges]
#         tgt = [sec_map[r['tgt']] for r in res_edges]
        
#         # Linear Scale
#         raw_weights = np.array([float(r['w']) for r in res_edges])
#         scaled_weights = raw_weights / max_val
        
#         data = HeteroData()
#         data['Sector'].x = x_sec
#         data['Sector'].y = y_sec
#         data['Sector', 'INPUT_TO', 'Sector'].edge_index = torch.tensor([src, tgt], dtype=torch.long)
#         data['Sector', 'INPUT_TO', 'Sector'].edge_weight = torch.tensor(scaled_weights, dtype=torch.float)
        
#         # Countries (Dummy)
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
#     model.load_state_dict(checkpoint['model'])
#     model.eval()
#     return model, data, df_sec

# def simulate_shock(target_uid, model, data, df_sec):
#     print(f"\n[SIMULATION] Targeting {target_uid}...")
    
#     # 1. Baseline
#     weight_dict_base = {('Sector', 'INPUT_TO', 'Sector'): data['Sector', 'INPUT_TO', 'Sector'].edge_weight}
#     with torch.no_grad():
#         y_base, _ = model(data.x_dict, data.edge_index_dict, weight_dict_base)
#         val_base = torch.expm1(y_base).numpy().flatten()

#     # 2. Intervention
#     if target_uid not in df_sec['id'].values:
#         print(f"   ❌ {target_uid} not found.")
#         return None

#     target_idx = df_sec[df_sec['id'] == target_uid].index[0]
#     edge_index = data['Sector', 'INPUT_TO', 'Sector'].edge_index
#     edge_weight = data['Sector', 'INPUT_TO', 'Sector'].edge_weight.clone()
    
#     # Identify edges connected to target
#     mask = (edge_index[0] == target_idx) | (edge_index[1] == target_idx)
    
#     # Apply 40% Blockade
#     edge_weight[mask] = edge_weight[mask] * 0.6 
    
#     # 3. Predict Shock
#     weight_dict_shock = {('Sector', 'INPUT_TO', 'Sector'): edge_weight}
#     with torch.no_grad():
#         y_shock, _ = model(data.x_dict, data.edge_index_dict, weight_dict_shock)
#         val_shock = torch.expm1(y_shock).numpy().flatten()

#     # 4. Calculate Loss
#     df_res = df_sec.copy()
#     df_res['loss'] = val_base - val_shock
    
#     # Filter victims (External only)
#     target_country = target_uid.split('_')[0]
#     ripples = df_res[(df_res['country'] != target_country) & (df_res['loss'] > 0.005)].sort_values('loss', ascending=False)
#     return ripples[['id', 'loss']]

# def run_comparison():
#     print("🧪 STARTING GEOPOLITICAL SCENARIO COMPARISON...")
#     model, data, df_sec = load_inference_system()
#     if not model: return

#     # Scenario A: China Electronics Shock
#     res_china = simulate_shock("CHN_C26", model, data, df_sec)
    
#     # Scenario B: Taiwan Electronics Shock
#     res_taiwan = simulate_shock("TWN_C26", model, data, df_sec)
    
#     print("\n=======================================================")
#     print(" 🇨🇳 SCENARIO A: CHINA SHOCK VICTIMS (Top 5)")
#     print("=======================================================")
#     if res_china is not None: print(res_china.head(5))
    
#     print("\n=======================================================")
#     print(" 🇹🇼 SCENARIO B: TAIWAN SHOCK VICTIMS (Top 5)")
#     print("=======================================================")
#     if res_taiwan is not None: print(res_taiwan.head(5))
    
#     print("\n✅ Comparison Complete. Use this table for your Discussion section.")

# if __name__ == "__main__":
#     run_comparison()
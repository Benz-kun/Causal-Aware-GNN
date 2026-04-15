#Version 2 for latest code

import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.express as px
from db_connection import get_driver, close_driver
from torch_geometric.data import HeteroData
from model import CausalHGT

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Neuro-Symbolic Trade Sanction Simulation", layout="wide", page_icon="🌍")

# --- 0. DICTIONARIES FOR HUMAN READABILITY ---
COUNTRY_MAP = {
    'AUS': '🇦🇺 Australia', 'AUT': '🇦🇹 Austria', 'BEL': '🇧🇪 Belgium', 'BGR': '🇧🇬 Bulgaria',
    'BRA': '🇧🇷 Brazil', 'CAN': '🇨🇦 Canada', 'CHE': '🇨🇭 Switzerland', 'CHN': '🇨🇳 China',
    'CYP': '🇨🇾 Cyprus', 'CZE': '🇨🇿 Czech Republic', 'DEU': '🇩🇪 Germany', 'DNK': '🇩🇰 Denmark',
    'ESP': '🇪🇸 Spain', 'EST': '🇪🇪 Estonia', 'FIN': '🇫🇮 Finland', 'FRA': '🇫🇷 France',
    'GBR': '🇬🇧 United Kingdom', 'GRC': '🇬🇷 Greece', 'HRV': '🇭🇷 Croatia', 'HUN': '🇭🇺 Hungary',
    'IDN': '🇮🇩 Indonesia', 'IND': '🇮🇳 India', 'IRL': '🇮🇪 Ireland', 'ITA': '🇮🇹 Italy',
    'JPN': '🇯🇵 Japan', 'KOR': '🇰🇷 South Korea', 'LTU': '🇱🇹 Lithuania', 'LUX': '🇱🇺 Luxembourg',
    'LVA': '🇱🇻 Latvia', 'MEX': '🇲🇽 Mexico', 'MLT': '🇲🇹 Malta', 'NLD': '🇳🇱 Netherlands',
    'NOR': '🇳🇴 Norway', 'POL': '🇵🇱 Poland', 'PRT': '🇵🇹 Portugal', 'ROU': '🇷🇴 Romania',
    'RUS': '🇷🇺 Russia', 'SVK': '🇸🇰 Slovakia', 'SVN': '🇸🇮 Slovenia', 'SWE': '🇸🇪 Sweden',
    'TUR': '🇹🇷 Turkey', 'TWN': '🇹🇼 Taiwan', 'USA': '🇺🇸 United States', 'ROW': '🌐 Rest of World'
}

# WIOD Sector Codes to Human Names
SECTOR_MAP = {
    'A01': 'Agriculture & Farming', 'A02': 'Forestry', 'A03': 'Fishing',
    'B': 'Mining & Quarrying',
    'C10-C12': 'Food, Beverages & Tobacco',
    'C13-C15': 'Textiles, Apparel & Leather',
    'C16': 'Wood Products (No Furniture)',
    'C17': 'Paper Products', 'C18': 'Printing & Media',
    'C19': 'Refined Petroleum & Coke',
    'C20': 'Chemicals', 'C21': 'Pharmaceuticals',
    'C22': 'Rubber & Plastics',
    'C23': 'Non-Metallic Minerals (Glass/Concrete)',
    'C24': 'Basic Metals (Steel/Aluminum)',
    'C25': 'Fabricated Metal Products',
    'C26': 'Electronics, Computers & Optical',
    'C27': 'Electrical Equipment',
    'C28': 'Machinery (Industrial)',
    'C29': 'Automotive (Vehicles)',
    'C30': 'Other Transport (Ships/Planes)',
    'C31_C32': 'Furniture & Manufacturing',
    'C33': 'Repair & Installation',
    'D35': 'Electricity & Gas Supply',
    'E36': 'Water Supply',
    'F': 'Construction',
    'G45': 'Car Sales & Repair',
    'G46': 'Wholesale Trade', 'G47': 'Retail Trade',
    'H49': 'Land Transport', 'H50': 'Water Transport', 'H51': 'Air Transport',
    'J58': 'Publishing', 'J61': 'Telecoms',
    'J62_J63': 'IT Services & Software',
    'K64': 'Financial Services', 'K65': 'Insurance',
    'L68': 'Real Estate',
    'M69_M70': 'Legal & Consulting',
    'M71': 'Engineering & Architecture',
    'M72': 'R&D (Scientific)',
    'Q': 'Healthcare', 'P85': 'Education'
}

def get_country_name(code):
    return COUNTRY_MAP.get(code, f"🏳️ {code}")

def get_sector_name(raw_name):
    # Try to match key WIOD codes if present in string
    for code, name in SECTOR_MAP.items():
        if code in raw_name:
            return f"{name} ({code})"
    return raw_name # Fallback

# --- 1. SYSTEM LOADER ---
@st.cache_resource
def load_system():
    try:
        checkpoint = torch.load("models/hetero_model.pth")
        max_val = checkpoint['max_val']
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    except FileNotFoundError:
        st.error("❌ Model missing! Run src/5_train_hetero.py first.")
        return None, None, None, None

    driver = get_driver()
    with driver.session() as session:
        # Load Nodes
        res_sec = session.run("""
            MATCH (s:Sector) 
            RETURN s.uid as id, s.name as name, s.country as country, s.gross_output as y
        """).data()
        df_sec = pd.DataFrame(res_sec)
        
        # --- ENRICH DATAFRAME WITH HUMAN NAMES ---
        df_sec['country_label'] = df_sec['country'].apply(get_country_name)
        df_sec['sector_label'] = df_sec['id'].apply(lambda x: get_sector_name(x.split('_')[1] if '_' in x else x))
        
        sec_map = {uid: i for i, uid in enumerate(df_sec['id'])}
        x_sec = torch.tensor(pd.get_dummies(df_sec['name']).values, dtype=torch.float)
        y_sec = torch.tensor(np.log1p(df_sec['y'].fillna(0).values), dtype=torch.float).view(-1, 1)

        # Load Edges
        res_edges = session.run("MATCH (s1:Sector)-[r:INPUT_TO]->(s2:Sector) RETURN s1.uid as src, s2.uid as tgt, r.weight as w").data()
        src = [sec_map[r['src']] for r in res_edges]
        tgt = [sec_map[r['tgt']] for r in res_edges]
        
        raw_weights = np.array([float(r['w']) for r in res_edges])
        scaled_weights = raw_weights / max_val
        
        data = HeteroData()
        data['Sector'].x = x_sec
        data['Sector'].y = y_sec
        data['Sector', 'INPUT_TO', 'Sector'].edge_index = torch.tensor([src, tgt], dtype=torch.long)
        data['Sector', 'INPUT_TO', 'Sector'].edge_weight = torch.tensor(scaled_weights, dtype=torch.float)
        
        # Dummy Country
        res_cou = session.run("MATCH (c:Country) RETURN c.iso_code as id").data()
        df_cou = pd.DataFrame(res_cou)
        data['Country'].x = torch.ones((len(df_cou), 1))
        
        # Loc Edges
        res_loc = session.run("MATCH (s:Sector)-[:LOCATED_IN]->(c:Country) RETURN s.uid as src, c.iso_code as tgt").data()
        cou_map = {uid: i for i, uid in enumerate(df_cou['id'])}
        l_src = [sec_map[r['src']] for r in res_loc]
        l_tgt = [cou_map[r['tgt']] for r in res_loc]
        data['Sector', 'LOCATED_IN', 'Country'].edge_index = torch.tensor([l_src, l_tgt], dtype=torch.long)

    close_driver(driver)
    
    model = CausalHGT(64, 1, data.metadata())
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, data, df_sec, max_val

# --- 2. SIMULATION ---
def run_simulation(model, data, df_sec, target_id, shock_pct):
    weight_dict_base = {('Sector', 'INPUT_TO', 'Sector'): data['Sector', 'INPUT_TO', 'Sector'].edge_weight}
    with torch.no_grad():
        y_base, _ = model(data.x_dict, data.edge_index_dict, weight_dict_base)
        val_base = torch.expm1(y_base).numpy().flatten()

    target_idx = df_sec[df_sec['id'] == target_id].index[0]
    edge_index = data['Sector', 'INPUT_TO', 'Sector'].edge_index
    edge_weight = data['Sector', 'INPUT_TO', 'Sector'].edge_weight.clone()
    
    mask = (edge_index[0] == target_idx) | (edge_index[1] == target_idx)
    factor = 1.0 - (shock_pct / 100.0)
    edge_weight[mask] = edge_weight[mask] * factor
    
    weight_dict_shock = {('Sector', 'INPUT_TO', 'Sector'): edge_weight}
    with torch.no_grad():
        y_shock, _ = model(data.x_dict, data.edge_index_dict, weight_dict_shock)
        val_shock = torch.expm1(y_shock).numpy().flatten()

    df_res = df_sec.copy()
    df_res['loss'] = val_base - val_shock
    target_country = target_id.split('_')[0]
    df_res = df_res[df_res['country'] != target_country]
    
    return df_res.sort_values('loss', ascending=False)

# --- 3. UI LAYOUT ---
st.title("🌐 Neuro-Symbolic Trade War Room")
st.markdown("### Structural Interference Engine")

model, data, df_sec, max_val = load_system()

if model:
    st.sidebar.header("🚀 Launch Scenario")
    
    # A. Country Selector (Human Readable)
    # We create a list of names, but map back to codes
    country_options = df_sec[['country', 'country_label']].drop_duplicates().sort_values('country_label')
    sel_country_label = st.sidebar.selectbox("Aggressor Country", country_options['country_label'])
    sel_country_code = country_options[country_options['country_label'] == sel_country_label]['country'].values[0]
    
    # B. Sector Selector (Human Readable)
    sector_options = df_sec[df_sec['country'] == sel_country_code][['id', 'sector_label']].sort_values('sector_label')
    sel_sector_label = st.sidebar.selectbox("Target Sector", sector_options['sector_label'])
    sel_sector_id = sector_options[sector_options['sector_label'] == sel_sector_label]['id'].values[0]
    
    shock_pct = st.sidebar.slider("Blockade Intensity (%)", 0, 100, 40)
    
    if st.sidebar.button("🔴 EXECUTE SANCTION"):
        with st.spinner(f"Simulating blockade on {sel_sector_label}..."):
            df_results = run_simulation(model, data, df_sec, sel_sector_id, shock_pct)
        
        col1, col2, col3 = st.columns(3)
        top_victim = df_results.iloc[0]
        total_loss = df_results[df_results['loss'] > 0]['loss'].sum()
        
        col1.metric("Global Impact Score", f"{total_loss:,.4f}")
        col2.metric("Primary Victim", f"{top_victim['country_label']}")
        col3.metric("Victim Loss", f"{top_victim['loss']:.5f}")
        
        st.subheader("🗺️ Global Contagion Heatmap")
        df_map = df_results.groupby('country')['loss'].sum().reset_index()
        # Map requires raw 3-letter codes
        fig = px.choropleth(df_map, locations="country", locationmode="ISO-3", color="loss",
                            hover_name="country", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Forensic Impact Report")
        # Display Human Readable Columns
        show_cols = ['country_label', 'sector_label', 'loss']
        st.dataframe(df_results[show_cols].head(20).style.format({'loss': '{:.6f}'}), use_container_width=True)

#Version 1 for old code

# import streamlit as st
# import pandas as pd
# import torch
# import numpy as np
# import os
# from model import TradeSAGE

# # --- CONFIGURATION ---
# # We point to the files we JUST created
# NODES_PATH = os.path.join("data", "processed", "nodes.pkl")
# EDGES_PATH = os.path.join("data", "processed", "edges.csv")
# MODEL_PATH = os.path.join("models", "trade_predictor.pth")

# st.set_page_config(page_title="Global Trade Ripple Simulator", layout="wide")

# @st.cache_resource
# def load_system():
#     # 1. Load Data
#     df_nodes = pd.read_pickle(NODES_PATH)
#     df_edges = pd.read_csv(EDGES_PATH)
    
#     # 2. Reconstruct the Graph Tensor (Same as training)
#     features = np.stack(df_nodes['embedding'].values)
#     x = torch.tensor(features, dtype=torch.float)
    
#     # Map UIDs to indices
#     uid_to_idx = {uid: i for i, uid in enumerate(df_nodes['uid'])}
#     src = [uid_to_idx[u] for u in df_edges['source']]
#     dst = [uid_to_idx[u] for u in df_edges['target']]
#     edge_index = torch.tensor([src, dst], dtype=torch.long)
    
#     # 3. Load the AI Model
#     model = TradeSAGE(in_channels=128, hidden_channels=64, out_channels=1)
#     model.load_state_dict(torch.load(MODEL_PATH))
#     model.eval()
    
#     return df_nodes, x, edge_index, model, uid_to_idx

# # Initialize System
# try:
#     df_nodes, x_base, edge_index, model, uid_to_idx = load_system()
#     st.success("✅ AI Brain Online: Hybrid SCM Connected")
# except Exception as e:
#     st.error(f"System Offline: {e}")
#     st.stop()

# # --- INTERFACE ---
# st.title("🌍 AI Trade War Simulator")
# st.markdown("Predict how sanctions ripple through the global economy using Graph Neural Networks.")

# # Sidebar Controls
# st.sidebar.header("⚡ Shock Configuration")

# # 1. Select Country
# countries = sorted(df_nodes['country'].unique())
# target_country = st.sidebar.selectbox("Target Country", countries, index=countries.index("CHN") if "CHN" in countries else 0)

# # 2. Select Sector
# country_nodes = df_nodes[df_nodes['country'] == target_country]
# sectors = sorted(country_nodes['name'].unique())
# target_sector_name = st.sidebar.selectbox("Target Sector", sectors)

# # Locate the Node
# target_node_row = country_nodes[country_nodes['name'] == target_sector_name].iloc[0]
# target_uid = target_node_row['uid']
# target_idx = uid_to_idx[target_uid]

# # 3. Apply Shock
# shock_pct = st.sidebar.slider("Sanction Intensity (%)", 0, 100, 50)
# st.sidebar.info(f"Simulating a **{shock_pct}% drop** in output for **{target_uid}**.")

# if st.sidebar.button("🔴 RUN SIMULATION"):
#     with st.spinner("Calculating Ripples..."):
#         # A. Baseline Prediction (Before Shock)
#         with torch.no_grad():
#             pred_base = torch.exp(model(x_base, edge_index)).numpy().flatten()
        
#         # B. Shocked Prediction (After Shock)
#         x_shocked = x_base.clone()
#         # Mathematically dampen the sector's embedding influence
#         shock_factor = 1.0 - (shock_pct / 100.0)
#         x_shocked[target_idx] = x_shocked[target_idx] * shock_factor
        
#         with torch.no_grad():
#             pred_shock = torch.exp(model(x_shocked, edge_index)).numpy().flatten()
            
#         # C. Calculate Impact
#         impact = pred_shock - pred_base
        
#         # D. Display Results
#         df_res = df_nodes.copy()
#         df_res['Loss ($M)'] = impact
#         df_res['Loss (%)'] = (impact / pred_base) * 100
        
#         # Filter: Don't show the target itself, show the victims
#         df_ripples = df_res[df_res['uid'] != target_uid].sort_values(by='Loss ($M)', ascending=True)
        
#         # Metrics
#         st.header(f"📉 Impact Analysis: {target_uid}")
#         col1, col2, col3 = st.columns(3)
#         col1.metric("Global Loss", f"${df_ripples['Loss ($M)'].sum():,.0f} M")
#         col2.metric("Top Victim", df_ripples.iloc[0]['country'])
#         col3.metric("Sectors Hit", f"{len(df_ripples[df_ripples['Loss ($M)'] < -0.1])}")

#         st.subheader("Top 10 Most Affected Sectors")
#         st.dataframe(df_ripples[['uid', 'name', 'Loss ($M)', 'Loss (%)']].head(10).style.format({'Loss ($M)': '${:,.2f}', 'Loss (%)': '{:.2f}%'}))
        
#         st.subheader("Global Impact Distribution")
#         st.bar_chart(df_ripples.head(50).set_index('uid')['Loss ($M)'])
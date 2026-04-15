import pandas as pd
import numpy as np
import os
from db_connection import get_driver, close_driver

# --- CONFIGURATION ---
OUTPUT_DIR = os.path.join("data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def export_training_data(driver):
    print("🚀 Starting Data Export for AI Training...")
    
    with driver.session() as session:
        # 1. Fetch Node Features (Embeddings + Economic Health + Metadata)
        print("   Fetching Node Features...")
        # We explicitly fetch 'country' and 'name' for the dashboard
        query_nodes = """
        MATCH (s:Sector)
        RETURN s.uid as uid, 
               s.country as country,
               s.name as name,
               s.embedding as embedding, 
               s.gross_output as gross_output
        """
        nodes = session.run(query_nodes).data()
        df_nodes = pd.DataFrame(nodes)
        
        # Fill missing values to prevent crashes
        df_nodes['gross_output'] = df_nodes['gross_output'].fillna(0)
        df_nodes['country'] = df_nodes['country'].fillna("Unknown")
        df_nodes['name'] = df_nodes['name'].fillna("Unknown")
        
        print(f"   ✅ Retrieved {len(df_nodes)} sectors.")
        
        # 2. Fetch Relationships
        print("   Fetching Trade Relationships...")
        query_edges = """
        MATCH (s1:Sector)-[r:INPUT_TO]->(s2:Sector)
        RETURN s1.uid as source, s2.uid as target, r.weight as weight
        """
        edges = session.run(query_edges).data()
        df_edges = pd.DataFrame(edges)
        
        print(f"   ✅ Retrieved {len(df_edges)} trade links.")

    # 3. Save to Disk
    # We pickle nodes because 'embedding' is a list/array object
    df_nodes.to_pickle(os.path.join(OUTPUT_DIR, "nodes.pkl"))
    df_edges.to_csv(os.path.join(OUTPUT_DIR, "edges.csv"), index=False)
    
    print(f"💾 Saved processed data to {OUTPUT_DIR}")

if __name__ == "__main__":
    driver = get_driver()
    if driver:
        export_training_data(driver)
        close_driver(driver)

# import pandas as pd
# import numpy as np
# import os
# from db_connection import get_driver, close_driver

# # --- CONFIGURATION ---
# OUTPUT_DIR = os.path.join("data", "processed")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def export_training_data(driver):
#     print("🚀 Starting Data Export for AI Training...")
    
#     with driver.session() as session:
#         # 1. Fetch Node Features (Embeddings + Economic Health)
#         print("   Fetching Node Features (Embeddings + Eco Data)...")
#         query_nodes = """
#         MATCH (s:Sector)
#         RETURN s.uid as uid, 
#                s.embedding as embedding, 
#                s.gross_output as gross_output, 
#                s.value_added as value_added, 
#                s.employment as employment
#         """
#         nodes = session.run(query_nodes).data()
#         df_nodes = pd.DataFrame(nodes)
        
#         # Parse embedding string/list if necessary (Neo4j returns list of floats)
#         # We also fill missing economic data with 0 to prevent AI crashes
#         df_nodes.fillna(0, inplace=True)
        
#         print(f"   ✅ Retrieved {len(df_nodes)} sectors.")
        
#         # 2. Fetch Relationships (The Supply Chain Edges)
#         print("   Fetching Trade Relationships...")
#         query_edges = """
#         MATCH (s1:Sector)-[r:INPUT_TO]->(s2:Sector)
#         RETURN s1.uid as source, s2.uid as target, r.weight as weight
#         """
#         edges = session.run(query_edges).data()
#         df_edges = pd.DataFrame(edges)
        
#         print(f"   ✅ Retrieved {len(df_edges)} trade links.")

#     # 3. Save to CSV
#     nodes_path = os.path.join(OUTPUT_DIR, "nodes.csv")
#     edges_path = os.path.join(OUTPUT_DIR, "edges.csv")
    
#     # We pickle the nodes because the 'embedding' column is a list/array
#     # CSVs struggle with lists inside cells. Pickle is faster for Python-to-Python.
#     df_nodes.to_pickle(os.path.join(OUTPUT_DIR, "nodes.pkl"))
#     df_edges.to_csv(edges_path, index=False)
    
#     print(f"💾 Saved processed data to:")
#     print(f"   - {os.path.join(OUTPUT_DIR, 'nodes.pkl')}")
#     print(f"   - {edges_path}")

# if __name__ == "__main__":
#     driver = get_driver()
#     if driver:
#         export_training_data(driver)
#         close_driver(driver)
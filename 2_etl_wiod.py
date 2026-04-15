#VERSION 3 CODE BELOW

import pandas as pd
import os
from db_connection import get_driver, close_driver
from tqdm import tqdm

# --- CONFIGURATION ---
FILE_PATH = os.path.join("data", "raw", "WIOT2014_Nov16_ROW.xlsb")
MIN_TRADE_VALUE = 5.0  # Million USD

SECTOR_MAP = {
    "A01": "Agriculture", "A02": "Forestry", "A03": "Fishing",
    "B": "Mining", "C10-C12": "Food_Beverages", "C13-C15": "Textiles",
    "C19": "Petroleum", "C20": "Chemicals", "C21": "Pharmaceuticals",
    "C26": "Electronics", "C29": "Automotive", "D35": "Energy",
    "F": "Construction", "H49": "Transport_Land", "H50": "Transport_Water",
    "H51": "Transport_Air", "J62_J63": "IT_Services", "K64": "Banking"
}

def get_sector_name(code):
    c = str(code).strip()
    if c in SECTOR_MAP: return SECTOR_MAP[c]
    for k in SECTOR_MAP:
        if c.startswith(k): return SECTOR_MAP[k]
    return c

def load_wiod_structure(driver):
    print(f"📂 Reading WIOD Matrix: {FILE_PATH}...")
    try:
        df = pd.read_excel(FILE_PATH, engine='pyxlsb', header=None)
    except FileNotFoundError:
        print(f"❌ File not found: {FILE_PATH}")
        return

    target_countries = df.iloc[4].ffill()
    target_codes = df.iloc[2]
    valid_rows = df.iloc[6:2470] 

    print("🚀 Building Heterogeneous Graph (Sector + Country)...")
    
    with driver.session() as session:
        batch_rels = []
        BATCH_SIZE = 5000 
        total_pushed = 0

        for idx, row in tqdm(valid_rows.iterrows(), total=len(valid_rows)):
            src_code = row[0]
            src_country = row[2]
            
            if pd.isna(src_country) or pd.isna(src_code): continue
            
            src_name = get_sector_name(src_code)
            src_uid = f"{str(src_country).strip()}_{str(src_code).strip()}"

            max_col = len(df.columns)
            for col_idx in range(5, max_col):
                val = row[col_idx]
                if pd.isna(val) or val < MIN_TRADE_VALUE: continue
                
                tgt_country = target_countries[col_idx]
                tgt_code = target_codes[col_idx]
                if pd.isna(tgt_country): continue
                
                tgt_name = get_sector_name(tgt_code)
                tgt_uid = f"{str(tgt_country).strip()}_{str(tgt_code).strip()}"

                batch_rels.append({
                    "src_uid": src_uid, "src_iso": str(src_country).strip(), "src_sec": src_name,
                    "tgt_uid": tgt_uid, "tgt_iso": str(tgt_country).strip(), "tgt_sec": tgt_name,
                    "weight": float(val)
                })

                if len(batch_rels) >= BATCH_SIZE:
                    _push_batch(session, batch_rels)
                    total_pushed += len(batch_rels)
                    batch_rels = []
        
        if batch_rels:
            _push_batch(session, batch_rels)
            total_pushed += len(batch_rels)
            
    print(f"✅ Structure Loaded. Total Trade Links Created: {total_pushed}")

def _push_batch(session, batch):
    query = """
    UNWIND $batch as row
    MERGE (c1:Country {iso_code: row.src_iso})
    MERGE (c2:Country {iso_code: row.tgt_iso})
    
    MERGE (s1:Sector {uid: row.src_uid})
    SET s1.name = row.src_sec, s1.country = row.src_iso
    
    MERGE (s2:Sector {uid: row.tgt_uid})
    SET s2.name = row.tgt_sec, s2.country = row.tgt_iso
    
    MERGE (s1)-[:LOCATED_IN]->(c1)
    MERGE (s2)-[:LOCATED_IN]->(c2)
    
    MERGE (s1)-[r:INPUT_TO]->(s2)
    SET r.weight = row.weight, r.year = 2014
    """
    session.run(query, batch=batch)

def verify_data(driver):
    print("\n🔍 VERIFICATION: Checking Graph Topology...")
    with driver.session() as session:
        # Check 1: Count Nodes
        r1 = session.run("MATCH (n) RETURN labels(n) as Type, count(n) as Count").data()
        print("   Node Counts:", r1)
        
        # Check 2: Validate Hierarchy (Does every Sector belong to a Country?)
        r2 = session.run("MATCH (s:Sector) WHERE NOT (s)-[:LOCATED_IN]->(:Country) RETURN count(s) as Orphans").single()
        print(f"   Orphan Sectors (Should be 0): {r2['Orphans']}")
        
        # Check 3: Sample a Trade Path
        r3 = session.run("""
            MATCH (c1:Country)<-[:LOCATED_IN]-(s1:Sector)-[r:INPUT_TO]->(s2:Sector)-[:LOCATED_IN]->(c2:Country)
            RETURN c1.iso_code + ' ' + s1.name + ' -> ' + c2.iso_code + ' ' + s2.name as Path, r.weight as Value
            LIMIT 1
        """).single()
        if r3:
            print(f"   Sample Flow: {r3['Path']} (${r3['Value']:.2f}M)")
        else:
            print("   ❌ No trade paths found!")

if __name__ == "__main__":
    driver = get_driver()
    if driver:
        # clear_database(driver) # Optional: Uncomment if you want to wipe before loading
        load_wiod_structure(driver)
        verify_data(driver)
        close_driver(driver)

#VERSION 2 CODE BELOW

# import pandas as pd
# import os
# from db_connection import get_driver, close_driver
# from tqdm import tqdm

# # --- CONFIGURATION ---
# FILE_PATH = os.path.join("data", "raw", "WIOT2014_Nov16_ROW.xlsb")
# MIN_TRADE_VALUE = 5.0  # Million USD

# # WIOD Sector Code Mapping
# SECTOR_MAP = {
#     "A01": "Agriculture", "A02": "Forestry", "A03": "Fishing",
#     "B": "Mining", "C10-C12": "Food_Beverages", "C13-C15": "Textiles",
#     "C19": "Petroleum", "C20": "Chemicals", "C21": "Pharmaceuticals",
#     "C26": "Electronics", "C29": "Automotive", "D35": "Energy",
#     "F": "Construction", "H49": "Transport_Land", "H50": "Transport_Water",
#     "H51": "Transport_Air", "J62_J63": "IT_Services", "K64": "Banking"
# }

# def get_sector_name(code):
#     c = str(code).strip()
#     if c in SECTOR_MAP: return SECTOR_MAP[c]
#     for k in SECTOR_MAP:
#         if c.startswith(k): return SECTOR_MAP[k]
#     return c

# def clear_database(driver):
#     print("🧹 Cleaning up old/corrupt data...")
#     with driver.session() as session:
#         session.run("MATCH (n) DETACH DELETE n")

# def load_wiod_data(driver):
#     print(f"📂 Reading WIOD Matrix: {FILE_PATH}...")
    
#     try:
#         df = pd.read_excel(FILE_PATH, engine='pyxlsb', header=None)
#     except FileNotFoundError:
#         print(f"❌ File not found: {FILE_PATH}")
#         return

#     print(f"📊 Matrix Loaded. Shape: {df.shape}")

#     # Metadata Rows (Targets) - Based on your successful debug
#     # Row 4 (Index 4) is Target Country
#     # Row 2 (Index 2) is Target Sector Code
#     target_countries = df.iloc[4].ffill()
#     target_codes = df.iloc[2]
    
#     # Data Block
#     # Starts at Row 6. Data Columns start at Index 5.
#     valid_rows = df.iloc[6:2470] 
    
#     print("🚀 Starting Graph Ingestion (Final Fix)...")
    
#     with driver.session() as session:
#         # Re-create constraints
#         session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Sector) REQUIRE s.uid IS UNIQUE;")
#         session.run("CREATE INDEX IF NOT EXISTS FOR (s:Sector) ON (s.country);")

#         batch_rels = []
#         BATCH_SIZE = 5000 
        
#         for idx, row in tqdm(valid_rows.iterrows(), total=len(valid_rows)):
#             try:
#                 # --- FIXED INDICES HERE ---
#                 # Based on your Debug Output:
#                 # Col 0 = Sector Code (e.g. A01)
#                 # Col 2 = Country Code (e.g. AUS)
#                 src_code = row[0]     
#                 src_country = row[2]  
                
#                 if pd.isna(src_country) or pd.isna(src_code): continue
                
#                 src_name = get_sector_name(src_code)
#                 src_uid = f"{str(src_country).strip()}_{str(src_code).strip()}"
                
#                 # Iterate Targets (Data starts at Column 5)
#                 max_col = len(df.columns)
                
#                 for col_idx in range(5, max_col):
#                     val = row[col_idx]
                    
#                     if pd.isna(val) or val < MIN_TRADE_VALUE: continue
                    
#                     tgt_country = target_countries[col_idx]
#                     tgt_code = target_codes[col_idx]
                    
#                     if pd.isna(tgt_country): continue
                    
#                     tgt_name = get_sector_name(tgt_code)
#                     tgt_uid = f"{str(tgt_country).strip()}_{str(tgt_code).strip()}"
                    
#                     batch_rels.append({
#                         "src_uid": src_uid, "src_iso": str(src_country).strip(), "src_sec": src_name,
#                         "tgt_uid": tgt_uid, "tgt_iso": str(tgt_country).strip(), "tgt_sec": tgt_name,
#                         "weight": float(val)
#                     })
                    
#                     if len(batch_rels) >= BATCH_SIZE:
#                         _push_batch(session, batch_rels)
#                         batch_rels = []
#             except Exception:
#                 continue
                
#         if batch_rels:
#             _push_batch(session, batch_rels)

#     print("✅ WIOD Data Ingestion Complete.")

# def _push_batch(session, batch):
#     query = """
#     UNWIND $batch as row
#     MERGE (s1:Sector {uid: row.src_uid})
#     SET s1.country = row.src_iso, s1.name = row.src_sec
    
#     MERGE (s2:Sector {uid: row.tgt_uid})
#     SET s2.country = row.tgt_iso, s2.name = row.tgt_sec
    
#     MERGE (s1)-[r:INPUT_TO]->(s2)
#     SET r.weight = row.weight, r.year = 2014
#     """
#     session.run(query, batch=batch)

# if __name__ == "__main__":
#     driver = get_driver()
#     if driver:
#         clear_database(driver) # Wipes the bad data
#         load_wiod_data(driver)
#         close_driver(driver)

#VERSION 1 CODE BELOW

# import pandas as pd
# import os
# from db_connection import get_driver, close_driver
# from tqdm import tqdm  # Progress bar

# # --- CONFIGURATION ---
# # Updated to match your actual file name
# FILE_PATH = os.path.join("data", "raw", "WIOT2014_Nov16_ROW.xlsb")
# MIN_TRADE_VALUE = 5.0  # Million USD. Filter small trades to speed up loading.

# # WIOD Sector Code Mapping (NACE Rev. 2)
# SECTOR_MAP = {
#     "A01": "Agriculture", "A02": "Forestry", "A03": "Fishing",
#     "B": "Mining", "C10-C12": "Food_Beverages", "C13-C15": "Textiles",
#     "C19": "Petroleum", "C20": "Chemicals", "C21": "Pharmaceuticals",
#     "C26": "Electronics", "C29": "Automotive", "D35": "Energy_Supply",
#     "F": "Construction", "H49": "Land_Transport", "H50": "Water_Transport",
#     "H51": "Air_Transport", "J62_J63": "IT_Services", "K64": "Banking"
# }

# def get_sector_name(code):
#     """Translates cryptic WIOD codes to readable names"""
#     # 1. Safety: Convert to string and strip whitespace
#     code_str = str(code).strip()
    
#     # 2. Handle "nan" or empty strings
#     if code_str.lower() == 'nan' or not code_str:
#         return "Unknown_Sector"
        
#     # 3. Exact match
#     if code_str in SECTOR_MAP: 
#         return SECTOR_MAP[code_str]
    
#     # 4. Partial match (e.g., A01 matches A01tA02)
#     for key in SECTOR_MAP:
#         if code_str.startswith(key): 
#             return SECTOR_MAP[key]
            
#     return code_str  # Fallback: return the code itself

# def load_wiod_data(driver):
#     print(f"📂 Reading WIOD Matrix: {FILE_PATH}...")
#     print("   (This involves parsing a large Excel file. Please wait ~1-2 mins...)")
    
#     try:
#         # Read the Excel file. 
#         df = pd.read_excel(FILE_PATH, engine='pyxlsb', header=[0])
#     except FileNotFoundError:
#         print(f"❌ ERROR: File not found at {FILE_PATH}")
#         return

#     print(f"📊 Matrix Loaded. Shape: {df.shape}")
    
#     # Filter columns to only keep Country/Sector pairs
#     # WIOD columns usually have length > 3 (e.g. "AUS", "Mining") vs metadata cols like "RNr"
#     # We'll use a safer check later.
    
#     print("🚀 Starting Graph Ingestion (Sector -> Sector)...")
    
#     with driver.session() as session:
#         session.run("CREATE INDEX IF NOT EXISTS FOR (s:Sector) ON (s.uid)")
        
#         # Limit rows to the Inter-Industry block (first ~2464 rows)
#         valid_rows = df.iloc[:2464] 
        
#         batch_rels = []
#         BATCH_SIZE = 5000 
        
#         for idx, row in tqdm(valid_rows.iterrows(), total=len(valid_rows), desc="Processing Rows"):
#             try:
#                 # Column 0: Country Code (e.g., AUS)
#                 # Column 1: Sector Code (e.g., A01)
#                 src_country = row.iloc[0]
#                 src_code = row.iloc[1]
                
#                 # Check for bad data in source
#                 if pd.isna(src_country) or pd.isna(src_code): 
#                     continue
                
#                 src_name = get_sector_name(src_code)
#                 src_uid = f"{str(src_country).strip()}_{str(src_code).strip()}"
                
#                 # Iterate Targets (Columns start at index 4 in WIOD usually)
#                 # We check the column count carefully
#                 num_cols = len(df.columns)
                
#                 for col_idx in range(4, num_cols):
#                     trade_value = row.iloc[col_idx]
                    
#                     # Skip empty or small values
#                     if pd.isna(trade_value) or trade_value < MIN_TRADE_VALUE:
#                         continue
                    
#                     # We need to identify the TARGET Country/Sector from the Column Header
#                     # But pandas read_excel with header=[0] flattens multi-index.
#                     # In WIOD, the column order usually mirrors the row order.
#                     # So Column 4 corresponds to Row 0's sector, Col 5 to Row 1, etc.
                    
#                     # Map Column Index back to Row Index to get Target Metadata
#                     target_row_idx = col_idx - 4
                    
#                     if target_row_idx >= len(valid_rows):
#                         # We reached the "Final Demand" columns (Households, Gov), skip for now
#                         break
                        
#                     target_meta = valid_rows.iloc[target_row_idx]
#                     tgt_country = target_meta.iloc[0]
#                     tgt_code = target_meta.iloc[1]
                    
#                     if pd.isna(tgt_country): continue
                    
#                     tgt_name = get_sector_name(tgt_code)
#                     tgt_uid = f"{str(tgt_country).strip()}_{str(tgt_code).strip()}"
                    
#                     batch_rels.append({
#                         "src_uid": src_uid, "src_iso": str(src_country).strip(), "src_sec": src_name,
#                         "tgt_uid": tgt_uid, "tgt_iso": str(tgt_country).strip(), "tgt_sec": tgt_name,
#                         "weight": float(trade_value)
#                     })
                    
#                     if len(batch_rels) >= BATCH_SIZE:
#                         _push_batch(session, batch_rels)
#                         batch_rels = []

#             except Exception as e:
#                 # Catch row-level errors so one bad row doesn't kill the script
#                 # print(f"Skipping row {idx}: {e}")
#                 continue
        
#         if batch_rels:
#             _push_batch(session, batch_rels)

#     print("✅ WIOD Data Ingestion Complete.")

# def _push_batch(session, batch):
#     query = """
#     UNWIND $batch as row
#     MERGE (s1:Sector {uid: row.src_uid})
#     SET s1.country = row.src_iso, s1.name = row.src_sec
    
#     MERGE (s2:Sector {uid: row.tgt_uid})
#     SET s2.country = row.tgt_iso, s2.name = row.tgt_sec
    
#     MERGE (s1)-[r:INPUT_TO]->(s2)
#     SET r.weight = row.weight, r.year = 2014
#     """
#     try:
#         session.run(query, batch=batch)
#     except Exception as e:
#         print(f"⚠️ Batch Write Error: {e}")

# if __name__ == "__main__":
#     driver = get_driver()
#     if driver:
#         load_wiod_data(driver)
#         close_driver(driver)

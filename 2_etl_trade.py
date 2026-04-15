import pandas as pd
from db_connection import get_driver, close_driver
from mappings import get_sector
import os

# --- CONFIGURATION (Matched to your headers) ---
DATA_PATH = os.path.join("data", "raw", "trade_data_usa_2022.csv") 

# These match the headers you provided exactly
COL_REPORTER = "reporterISO"
COL_PARTNER = "partnerISO"
COL_HS = "cmdCode"
COL_VALUE = "primaryValue"
COL_YEAR = "refYear"

def load_trade_data(driver):
    print(f"📂 Reading CSV from: {DATA_PATH}")
    
    # 1. Read CSV
    try:
        # dtype={'cmdCode': str} forces the HS code to be read as a string "01" not number 1
        df = pd.read_csv(DATA_PATH, dtype={COL_HS: str, COL_REPORTER: str, COL_PARTNER: str})
    except FileNotFoundError:
        print("❌ CRITICAL ERROR: File not found. Check your file name in data/raw/")
        return

    print(f"📊 Found {len(df)} rows in CSV.")

    # 2. Open Neo4j Session
    with driver.session() as session:
        count = 0
        errors = 0
        
        for index, row in df.iterrows():
            # --- DATA CLEANING (The secret sauce) ---
            source_iso = row.get(COL_REPORTER)
            target_iso = row.get(COL_PARTNER)
            hs_code_raw = row.get(COL_HS)
            value_raw = row.get(COL_VALUE)
            year_raw = row.get(COL_YEAR)

            # Skip garbage rows (e.g. if ISO code is missing or NaN)
            if pd.isna(source_iso) or pd.isna(target_iso):
                if errors < 5: print(f"⚠️ Skipping Row {index}: Missing ISO Code")
                errors += 1
                continue

            # Force Data Types (Neo4j is strict!)
            sector = get_sector(hs_code_raw)  # Convert HS code to Sector Name
            
            try:
                params = {
                    "source_iso": str(source_iso).strip(),
                    "target_iso": str(target_iso).strip(),
                    "sector_name": str(sector),
                    "year": int(year_raw) if pd.notna(year_raw) else 2022,
                    "value": float(value_raw) if pd.notna(value_raw) else 0.0
                }
            except ValueError as e:
                if errors < 5: print(f"⚠️ Value Error on row {index}: {e}")
                errors += 1
                continue

            # --- DEBUG: Print the first valid row to ensure it looks right ---
            if count == 0:
                print("\n🔍 INSPECTING FIRST RECORD TO BE INSERTED:")
                print(f"   Source: {params['source_iso']} | Target: {params['target_iso']}")
                print(f"   Sector: {params['sector_name']} (from HS: {hs_code_raw})")
                print(f"   Value: ${params['value']:,}")
                print("-" * 50)

            # 3. The Cypher Query
            cypher = """
            MERGE (c1:Country {iso_code: $source_iso})
            MERGE (c2:Country {iso_code: $target_iso})
            MERGE (s:Sector {sector_id: $sector_name})
            
            MERGE (c1)-[r:TRADES_WITH {sector: $sector_name}]->(c2)
            ON CREATE SET r.weight = $value, r.year = $year
            ON MATCH SET r.weight = r.weight + $value
            """
            
            try:
                session.run(cypher, params)
                count += 1
                if count % 1000 == 0:
                    print(f"   🚀 Loaded {count} trades...")
            except Exception as e:
                print(f"❌ Neo4j Error on row {index}: {e}")
                break

    print(f"\n✅ ETL Process Finished.")
    print(f"   Successfully loaded: {count} relationships")
    print(f"   Skipped/Errors: {errors}")

if __name__ == "__main__":
    driver = get_driver()
    if driver:
        load_trade_data(driver)
        close_driver(driver)

# import pandas as pd
# from db_connection import get_driver, close_driver
# from mappings import get_sector
# import os

# # 1. Configuration
# DATA_PATH = os.path.join("data", "raw", "trade_data_usa_2022.csv") 

# # NOTE: Adjust these column names based on your specific CSV download!
# # Open your CSV and check exactly what the headers are.
# COL_REPORTER = "reporterISO"   # e.g., 'USA'
# COL_PARTNER = "partnerISO"     # e.g., 'CHN'
# COL_HS = "cmdCode"             # e.g., '85'
# COL_VALUE = "primaryValue"     # e.g., 100000 (The $ amount)
# COL_YEAR = "refYear"              # e.g., 2022

# def load_trade_data(driver):
#     print(f"📂 Loading data from {DATA_PATH}...")
    
#     # Read the CSV (using pandas)
#     try:
#         df = pd.read_csv(DATA_PATH)
#     except FileNotFoundError:
#         print("❌ Error: File not found! Did you download the CSV into data/raw/?")
#         return

#     print(f"📊 Found {len(df)} rows. Processing...")

#     query = """
#     MERGE (source:Country {iso_code: $source_iso})
#     MERGE (target:Country {iso_code: $target_iso})
    
#     MERGE (s:Sector {sector_id: $sector_name})
    
#     # Create the relationship: Country -> Sector -> Country
#     # This represents: USA exports Electronics to China
#     MERGE (source)-[r:EXPORTS_TO {year: $year}]->(target)
    
#     # We store the sector breakdown as a property on the edge OR as a separate node structure
#     # For this phase, let's keep it simple: Aggregate the weights.
#     SET r.total_value = coalesce(r.total_value, 0) + $value
#     """
    
#     # Neo4j Batch Import (More efficient)
#     with driver.session() as session:
#         count = 0
#         for index, row in df.iterrows():
            
#             # --- DEBUGGING BLOCK START ---
#             if index == 0:
#                 print("\n🔍 DEBUGGING FIRST ROW DATA:")
#                 print(f"Raw Row: {row.to_dict()}")
#                 print(f"Mapped Sector: {get_sector(row.get(COL_HS))}")
#                 print(f"Source ISO: '{row.get(COL_REPORTER)}' (Type: {type(row.get(COL_REPORTER))})")
#                 print(f"Target ISO: '{row.get(COL_PARTNER)}' (Type: {type(row.get(COL_PARTNER))})")
#             # --- DEBUGGING BLOCK END ---
            
#             # 1. Map HS Code to Sector
#             hs_code = row.get(COL_HS)
#             sector = get_sector(hs_code)
            
#             # 2. Prepare Data
#             params = {
#                 "source_iso": row.get(COL_REPORTER),
#                 "target_iso": row.get(COL_PARTNER),
#                 "sector_name": sector,
#                 "year": int(row.get(COL_YEAR, 2022)),
#                 "value": float(row.get(COL_VALUE, 0))
#             }
            
#             # Skip invalid rows
#             if not params["source_iso"] or not params["target_iso"]:
#                 continue

#             # 3. Run Query (We create a nuanced graph path here)
#             # Path: (Country A)-[:PRODUCES]->(Sector Node)-[:FLOWS_TO]->(Country B)
#             # This is better for Graph Neural Networks later.
            
#             cypher = """
#             MERGE (c1:Country {iso_code: $source_iso})
#             MERGE (c2:Country {iso_code: $target_iso})
#             MERGE (s:Sector {sector_id: $sector_name})
            
#             // Create a specific trade relationship for this sector
#             MERGE (c1)-[r:TRADES_WITH {sector: $sector_name}]->(c2)
#             ON CREATE SET r.weight = $value, r.year = $year
#             ON MATCH SET r.weight = r.weight + $value
#             """
            
#             session.run(cypher, params)
#             count += 1
            
#             if count % 100 == 0:
#                 print(f"   Processed {count} trades...")

#     print("✅ ETL Complete. Data is in Neo4j.")

# if __name__ == "__main__":
#     driver = get_driver()
#     if driver:
#         load_trade_data(driver)
#         close_driver(driver)
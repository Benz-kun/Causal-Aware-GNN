import pandas as pd
import os
from db_connection import get_driver, close_driver

# --- CONFIGURATION ---
# We use the .xlsx file you confirmed you have
FILE_PATH = os.path.join("data", "raw", "WIOD_SEA_Nov16.xlsx")
SHEET_NAME = "DATA"

# The specific economic indicators we need for the AI model
# GO = Gross Output (Total money made)
# VA = Value Added (Contribution to GDP)
# EMP = Employment (Number of people working)
VARS_TO_KEEP = ["GO", "VA", "EMP"]
TARGET_YEAR = 2014  # The column name in the Excel file is the integer 2014

def enrich_sectors(driver):
    print(f"📂 Reading Economic Health Data: {FILE_PATH}...")
    
    try:
        # Load the specific 'DATA' sheet
        df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        print("   Make sure 'WIOD_SEA_Nov16.xlsx' is in 'data/raw/'")
        return

    print(f"📊 Data Loaded. Shape: {df.shape}")
    
    # 1. Filter for the variables we want (GO, VA, EMP)
    # The column name in your file is lowercase 'variable'
    df_filtered = df[df['variable'].isin(VARS_TO_KEEP)].copy()
    
    print(f"   Filtered down to {len(df_filtered)} relevant rows.")

    print(f"🚀 Enriching Sector Nodes with {VARS_TO_KEEP} for Year {TARGET_YEAR}...")
    
    updates = []
    
    # 2. Iterate and prepare Graph Updates
    for idx, row in df_filtered.iterrows():
        country = row['country']
        code = row['code']
        variable = row['variable']
        value = row[TARGET_YEAR] # Access the '2014' column
        
        # Skip bad data
        if pd.isna(country) or pd.isna(code) or pd.isna(value):
            continue
            
        # Construct the UID exactly how we did in the Trade script
        # Format: "AUS_A01"
        uid = f"{str(country).strip()}_{str(code).strip()}"
        
        # Map variable codes to readable property names
        prop_name = "unknown"
        if variable == "GO": prop_name = "gross_output"
        elif variable == "VA": prop_name = "value_added"
        elif variable == "EMP": prop_name = "employment"
        
        updates.append({
            "uid": uid,
            "property": prop_name,
            "value": float(value)
        })

    # 3. Batch Update Neo4j
    with driver.session() as session:
        # Separate updates by property type to keep Cypher simple
        go_data = [x for x in updates if x['property'] == "gross_output"]
        va_data = [x for x in updates if x['property'] == "value_added"]
        emp_data = [x for x in updates if x['property'] == "employment"]
        
        _run_update(session, go_data, "gross_output")
        _run_update(session, va_data, "value_added")
        _run_update(session, emp_data, "employment")
        
    print("✅ Economic Confounders Added Successfully.")

def _run_update(session, data, property_name):
    if not data: return
    
    print(f"   Writing {len(data)} values for '{property_name}'...")
    
    # We construct a query that matches the Sector by UID and sets the specific property
    query = f"""
    UNWIND $batch as row
    MATCH (s:Sector {{uid: row.uid}})
    SET s.{property_name} = row.value
    """
    session.run(query, batch=data)

if __name__ == "__main__":
    driver = get_driver()
    if driver:
        enrich_sectors(driver)
        close_driver(driver)
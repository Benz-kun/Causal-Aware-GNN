#VERSION 2 CODE BELOW

import os
from db_connection import get_driver, close_driver

def reset_and_setup_schema(driver):
    print("⚠️ WARNING: This will WIPE the database to ensure a clean Research Schema.")
    confirm = input("Type 'YES' to proceed: ")
    if confirm != "YES":
        print("Operation aborted.")
        return

    with driver.session() as session:
        # 1. Clean Slate
        print("🧹 Wiping Database...")
        session.run("MATCH (n) DETACH DELETE n")
        
        # 2. Define Constraints (The "Heterogeneous" Structure)
        print("🏗️ Applying Research Constraints...")
        
        # Country: The Macro Unit (e.g., 'CHN', 'USA')
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Country) REQUIRE c.iso_code IS UNIQUE")
        
        # Sector: The Micro Unit (e.g., 'CHN_Electronics')
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Sector) REQUIRE s.uid IS UNIQUE")
        
        # Event: The Signal Unit (e.g., 'Sanction_2022_05')
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.url IS UNIQUE")

        # 3. Create Indexes for Speed
        session.run("CREATE INDEX IF NOT EXISTS FOR (s:Sector) ON (s.country)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.date)")

    print("✅ Schema Applied: Heterogeneous Graph Ready.")

if __name__ == "__main__":
    driver = get_driver()
    if driver:
        reset_and_setup_schema(driver)
        close_driver(driver)

#VERSION 1 CODE BELOW
# # File: src/1_schema_setup.py (UPDATED FOR RESEARCH PHASE)
# from db_connection import get_driver, close_driver

# def setup_constraints(driver):
#     queries = [
#         # 1. Country Nodes: Still needed as containers
#         "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Country) REQUIRE c.iso_code IS UNIQUE;",
        
#         # 2. Sector Nodes: The STARS of the show now.
#         # Format: "USA_Auto", "CHN_Electronics"
#         "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Sector) REQUIRE s.uid IS UNIQUE;",
        
#         # 3. Events: Same as before
#         "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE;",
        
#         # 4. New Indexes for Performance
#         "CREATE INDEX IF NOT EXISTS FOR (s:Sector) ON (s.country_iso);",
#         "CREATE INDEX IF NOT EXISTS FOR (s:Sector) ON (s.industry_name);"
#     ]
    
#     with driver.session() as session:
#         for q in queries:
#             print(f"Executing: {q}")
#             session.run(q)
#     print("✅ Research-Grade Schema (WIOD) applied successfully!")

# if __name__ == "__main__":
#     driver = get_driver()
#     if driver:
#         setup_constraints(driver)
#         close_driver(driver)
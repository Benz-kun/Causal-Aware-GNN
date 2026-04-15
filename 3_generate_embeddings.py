#CODE VERSION 1

import time
from db_connection import get_driver, close_driver

def generate_embeddings(driver):
    print("🚀 Generating Neuro-Symbolic Embeddings (FastRP)...")
    
    with driver.session() as session:
        # 1. Clean Slate: Remove old projections
        session.run("CALL gds.graph.drop('world_trade', false)")

        # 2. Project the Economic Graph
        # We only project Sectors and Trade flows to learn structural position
        print("   Projecting graph into memory...")
        session.run("""
            CALL gds.graph.project(
                'world_trade',
                'Sector',
                {
                    INPUT_TO: {
                        orientation: 'NATURAL',
                        properties: 'weight'
                    }
                }
            )
        """)
        
        # 3. Run FastRP (Random Projection)
        # This converts the "Trade Web" into a 128-dimensional vector
        print("   Calculating structural vectors...")
        start_time = time.time()
        
        session.run("""
            CALL gds.fastRP.write(
                'world_trade',
                {
                    embeddingDimension: 48,
                    relationshipWeightProperty: 'weight',
                    writeProperty: 'embedding'
                }
            )
        """)
        
        # 4. Cleanup
        session.run("CALL gds.graph.drop('world_trade', false)")
        
        duration = time.time() - start_time
        print(f"✅ Success! Generated embeddings for all sectors in {duration:.2f}s.")

if __name__ == "__main__":
    driver = get_driver()
    if driver:
        generate_embeddings(driver)
        close_driver(driver)

#OLD CODE VERSION 1

# import time
# from db_connection import get_driver, close_driver

# def generate_embeddings(driver):
#     print("🚀 Starting Graph Embedding Generation (FastRP)...")
    
#     with driver.session() as session:
#         # 1. Check if GDS is installed
#         try:
#             result = session.run("RETURN gds.version()").single()
#             print(f"✅ GDS Plugin Detected: Version {result[0]}")
#         except Exception:
#             print("❌ GDS Plugin NOT detected!")
#             return

#         # 2. Drop old graph projection if it exists
#         exists = session.run("CALL gds.graph.exists('world_trade') YIELD exists RETURN exists").single()[0]
#         if exists:
#             print("   Dropping old in-memory graph...")
#             session.run("CALL gds.graph.drop('world_trade')")

#         # 3. Create In-Memory Graph Projection
#         print("   Projecting graph into memory...")
#         session.run("""
#             CALL gds.graph.project(
#                 'world_trade',
#                 'Sector',
#                 {
#                     INPUT_TO: {
#                         orientation: 'NATURAL',
#                         properties: 'weight'
#                     }
#                 }
#             )
#         """)
        
#         # 4. Run FastRP (Fixed for GDS 2.x)
#         # CHANGED: 'propertiesWritten' -> 'nodePropertiesWritten'
#         print("   Calculating embeddings (this may take a moment)...")
#         start_time = time.time()
        
#         result = session.run("""
#             CALL gds.fastRP.write(
#                 'world_trade',
#                 {
#                     embeddingDimension: 128,
#                     relationshipWeightProperty: 'weight',
#                     writeProperty: 'embedding'
#                 }
#             )
#             YIELD nodeCount, nodePropertiesWritten
#         """).single()
        
#         duration = time.time() - start_time
#         print(f"✅ Embeddings Generated in {duration:.2f} seconds.")
#         print(f"   Nodes Processed: {result['nodeCount']}")
#         print(f"   Properties Written: {result['nodePropertiesWritten']}")

#         # 5. Clean up
#         session.run("CALL gds.graph.drop('world_trade')")
#         print("🧹 In-memory graph cleaned up.")

# if __name__ == "__main__":
#     driver = get_driver()
#     if driver:
#         generate_embeddings(driver)
#         close_driver(driver)
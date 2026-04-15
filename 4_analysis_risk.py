import pandas as pd
from db_connection import get_driver, close_driver

def calculate_risk_scores(driver):
    print("🧮 Calculating Geopolitical Risk Scores...")
    
    with driver.session() as session:
        # 1. Count events per country
        # We assume for this prototype that ANY news event implies volatility/risk.
        # In a real system, we would filter by negative tone.
        cypher = """
        MATCH (c:Country)<-[:IMPACTS]-(e:Event)
        WITH c, count(e) as event_count
        
        // Normalize score (simple version: 1 event = 10 risk points)
        WITH c, event_count, (event_count * 10) as risk_score
        
        // Write the score back to the Country node
        SET c.risk_score = risk_score
        
        RETURN c.iso_code as country, event_count, risk_score
        ORDER BY risk_score DESC
        """
        
        result = session.run(cypher)
        
        # Display results
        data = [record.data() for record in result]
        df = pd.DataFrame(data)
        
        if not df.empty:
            print("\n🔥 TOP RISK COUNTRIES (Based on News Volume):")
            print(df.head(10))
        else:
            print("⚠️ No risks calculated. Did you link events to countries other than USA?")

def propagate_risk_to_supply_chain(driver):
    print("\n🌊 Propagating Risk to Supply Chains...")
    
    with driver.session() as session:
        # 2. Who trades with high-risk countries?
        # If Country A has high risk, Country B (who imports from A) inherits some risk.
        cypher = """
        MATCH (supplier:Country)-[t:TRADES_WITH]->(buyer:Country)
        WHERE supplier.risk_score IS NOT NULL AND supplier.risk_score > 0
        
        RETURN 
            buyer.iso_code as Buying_Country,
            supplier.iso_code as Risky_Supplier,
            supplier.risk_score as Supplier_Risk,
            t.sector as Sector,
            t.weight as Trade_Value
        ORDER BY Supplier_Risk DESC, Trade_Value DESC
        LIMIT 10
        """
        
        result = session.run(cypher)
        data = [record.data() for record in result]
        df = pd.DataFrame(data)
        
        if not df.empty:
            print("\n🚨 SUPPLY CHAINS AT RISK:")
            print(df.to_string(index=False))
        else:
            print("⚠️ No supply chain risks found.")

if __name__ == "__main__":
    driver = get_driver()
    if driver:
        calculate_risk_scores(driver)
        propagate_risk_to_supply_chain(driver)
        close_driver(driver)
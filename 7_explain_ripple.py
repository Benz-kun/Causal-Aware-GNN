import torch
import pandas as pd
import numpy as np
from db_connection import get_driver, close_driver

def generate_forensic_report():
    print("🕵️ STARTING DEEP FORENSIC TRACING (Direct + Indirect)...")
    
    driver = get_driver()
    if not driver: return

    try:
        checkpoint = torch.load("models/hetero_model.pth")
        max_val = checkpoint['max_val']
    except:
        max_val = 1.0

    with driver.session() as session:
        SOURCE = "CHN_C26"
        # The victims identified by your validation script
        VICTIMS = ["LTU_C26", "HUN_C26", "CZE_C26", "TUR_C26"] 
        
        print(f"\n🔍 Tracing Contagion: {SOURCE} --> {VICTIMS}")
        
        for victim in VICTIMS:
            print(f"\n   ================================================")
            print(f"   [ANALYSIS] Target: {victim}")
            
            # 1. DIRECT EXPOSURE
            q_direct = f"""
            MATCH (source:Sector {{uid: '{SOURCE}'}})-[r:INPUT_TO]->(victim:Sector {{uid: '{victim}'}})
            RETURN r.weight as value
            """
            res_direct = session.run(q_direct).data()
            
            if res_direct:
                val = float(res_direct[0]['value'])
                print(f"   🚨 DIRECT LINK: {SOURCE} ==($ {val:,.0f})==> {victim}")
            else:
                print(f"   ℹ️  No Direct Link found.")

            # 2. INDIRECT EXPOSURE (The "Ripple" Path)
            # Find the biggest middlemen
            q_indirect = f"""
            MATCH (source:Sector {{uid: '{SOURCE}'}})-[r1:INPUT_TO]->(mid:Sector)-[r2:INPUT_TO]->(victim:Sector {{uid: '{victim}'}})
            WHERE mid.uid <> '{victim}' AND mid.uid <> '{SOURCE}'
            RETURN mid.uid as middleman, r1.weight as w1, r2.weight as w2, (r1.weight + r2.weight) as total_flow
            ORDER BY total_flow DESC LIMIT 3
            """
            res_indirect = session.run(q_indirect).data()
            
            if res_indirect:
                print(f"   ⚠️  INDIRECT CONTAGION (Top 3 Hubs):")
                for r in res_indirect:
                    w1, w2 = r['w1'], r['w2']
                    mid = r['middleman']
                    # Calculate 'BottleNeck' score: The flow is limited by the smaller of the two pipes
                    flow_strength = min(w1, w2)
                    print(f"      Path: {SOURCE} --($ {w1:,.0f})--> [{mid}] --($ {w2:,.0f})--> {victim}")
            else:
                print("      ❌ No obvious 2-hop path found.")

    close_driver(driver)
    print("\n✅ Deep Forensic Report Complete.")

if __name__ == "__main__":
    generate_forensic_report()
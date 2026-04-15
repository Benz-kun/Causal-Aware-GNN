#VERSION 3 CODE BELOW

import requests
import pandas as pd
from db_connection import get_driver, close_driver
from mappings import get_iso3 

# --- CONFIGURATION ---
# GDELT 2.0 Doc API - The largest open database of human society
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def fetch_real_gdelt_events():
    print("📡 Connecting to GDELT Global Knowledge Graph...")
    
    # We look back 3 months ('3mo') to ensure we catch significant structural shocks
    # tone<-5 filters for negative news (conflict/tension)
    params = {
        "query": '(sanction OR "trade war" OR tariff OR embargo) tone<-5',
        "mode": "ArtList",
        "maxrecords": "75",
        "timespan": "3mo", 
        "format": "json"
    }
    
    try:
        response = requests.get(GDELT_API_URL, params=params)
        data = response.json()
        
        if "articles" in data:
            print(f"✅ API Success: Retrieved {len(data['articles'])} real-world signals.")
            return data['articles']
        else:
            print("⚠️ API returned 0 results. Try widening the query.")
            return []
            
    except Exception as e:
        print(f"❌ API Connection Failed: {e}")
        return []

def load_events(driver, events):
    print("📂 Ingesting Events into Heterogeneous Graph...")
    with driver.session() as session:
        count = 0
        for event in events:
            # 1. Parse Event Data
            title = event.get('title', 'Unknown')
            url = event.get('url', '')
            date_str = event.get('seendate', '20240101')[:8] # Format: YYYYMMDD
            
            # 2. Heuristic: Identify Countries in the Title
            # In a full paper, we would use GDELT's 'locations' field, but title match is safer for now.
            involved_iso = []
            
            # Simple keyword matching against major economies
            if 'China' in title or 'Beijing' in title: involved_iso.append('CHN')
            if 'US' in title or 'United States' in title or 'Biden' in title: involved_iso.append('USA')
            if 'Russia' in title or 'Moscow' in title or 'Putin' in title: involved_iso.append('RUS')
            if 'EU' in title or 'Europe' in title or 'Germany' in title: involved_iso.append('DEU')
            
            # If no major powers involved, skip to keep noise low
            if not involved_iso: continue

            # 3. Create Graph Connections
            query = """
            MERGE (e:Event {url: $url})
            SET e.title = $title, e.date = $date, e.source = 'GDELT_Live'
            
            WITH e
            UNWIND $isos as iso
            MATCH (c:Country {iso_code: iso})
            MERGE (e)-[r:IMPACTS]->(c)
            SET r.weight = 1.0  // Default impact weight
            """
            
            session.run(query, url=url, title=title, date=date_str, isos=involved_iso)
            count += 1
            
    print(f"✅ Knowledge Graph Updated: Linked {count} events to countries.")

if __name__ == "__main__":
    events = fetch_real_gdelt_events()
    if events:
        driver = get_driver()
        if driver:
            load_events(driver, events)
            close_driver(driver)

#VERSION 2 CODE BELOW

# import requests
# import pandas as pd
# from db_connection import get_driver, close_driver
# from datetime import datetime

# # --- CONFIGURATION ---
# GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# def fetch_gdelt_events():
#     """
#     Fetches recent news events from GDELT 2.0 Doc API.
#     """
#     print("📡 Querying GDELT Project API...")
    
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
#     }

#     # Query: US + Trade/Sanctions
#     params = {
#         "query": '"United States" (trade OR economy OR sanction)',
#         "mode": "ArtList",
#         "maxrecords": "50",
#         "format": "json"
#     }
    
#     try:
#         response = requests.get(GDELT_API_URL, params=params, headers=headers)
#         if response.status_code != 200:
#             print(f"❌ HTTP Error {response.status_code}")
#             return []

#         try:
#             data = response.json()
#         except ValueError:
#             print("❌ Error: API returned non-JSON data.")
#             return []
        
#         if "articles" in data:
#             print(f"✅ Received {len(data['articles'])} events from GDELT.")
#             return data['articles']
#         else:
#             print("⚠️ No 'articles' key found.")
#             return []
            
#     except Exception as e:
#         print(f"❌ API Connection Error: {e}")
#         return []

# def load_events(driver, events):
#     """
#     Loads events into Neo4j and links them to Countries.
#     """
#     print("📂 Loading events into Neo4j...")
    
#     with driver.session() as session:
#         count = 0
        
#         for event in events:
#             # 1. Extract Data
#             url = event.get('url', '')
#             title = event.get('title', 'Unknown Event')
#             date_str = event.get('seendate', '20220101') 
            
#             # 2. Heuristic Linking
#             involved_countries = ['USA'] 
            
#             if 'China' in title or 'Chinese' in title: involved_countries.append('CHN')
#             if 'Russia' in title or 'Russian' in title: involved_countries.append('RUS')
#             if 'Germany' in title or 'German' in title: involved_countries.append('DEU')
#             if 'Japan' in title or 'Japanese' in title: involved_countries.append('JPN')
#             if 'Mexico' in title or 'Mexican' in title: involved_countries.append('MEX')
#             if 'UK' in title or 'Britain' in title: involved_countries.append('GBR')

#             # 3. Cypher Query (CORRECTED COMMENTS)
#             cypher = """
#             MERGE (e:Event {url: $url})
#             SET e.title = $title, 
#                 e.date = $date,
#                 e.source = 'GDELT'
            
#             // Create relationships for all involved countries
#             WITH e
#             UNWIND $countries AS iso_code
#             MATCH (c:Country {iso_code: iso_code})
#             MERGE (e)-[r:IMPACTS]->(c)
#             SET r.tone = -5.0 
#             """
            
#             try:
#                 session.run(cypher, {
#                     'url': url, 
#                     'title': title, 
#                     'date': date_str,
#                     'countries': involved_countries
#                 })
#                 count += 1
#             except Exception as e:
#                 print(f"⚠️ Error loading event: {e}")

#     print(f"✅ Successfully loaded {count} events into Neo4j.")

# if __name__ == "__main__":
#     events_data = fetch_gdelt_events()
    
#     if events_data:
#         driver = get_driver()
#         if driver:
#             load_events(driver, events_data)
#             close_driver(driver)


#VERSION 1 CODE BELOW

# import requests
# import pandas as pd
# from db_connection import get_driver, close_driver
# from datetime import datetime

# # --- CONFIGURATION ---
# # Correct GDELT 2.0 Doc API Endpoint
# GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# def fetch_gdelt_events():
#     """
#     Fetches recent news events from GDELT 2.0 Doc API.
#     """
#     print("📡 Querying GDELT Project API...")
    
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
#     }
    
#     params = {
#         "query": '"United States" (trade OR economy OR sanction)',
#         "mode": "ArtList",
#         "maxrecords": "50",
#         "format": "json"
#     }
    
#     try:
#         response = requests.get(GDELT_API_URL, params=params, headers=headers)
        
#         # Check for HTTP errors
#         if response.status_code != 200:
#             print(f"❌ HTTP Error {response.status_code}: {response.text[:200]}")
#             return []

#         # Parse JSON
#         try:
#             data = response.json()
#         except ValueError:
#             print("❌ Error: API returned non-JSON data.")
#             print(f"Raw Response Snippet: {response.text[:200]}") 
#             return []
        
#         if "articles" in data:
#             print(f"✅ Received {len(data['articles'])} events from GDELT.")
#             return data['articles']
#         else:
#             print("⚠️ No 'articles' key found. API might have returned 0 results.")
#             return []
            
#     except Exception as e:
#         print(f"❌ API Connection Error: {e}")
#         return []

# def load_events(driver, events):
#     """
#     Loads events into Neo4j and links them to Countries based on title mentions.
#     """
#     print("📂 Loading events into Neo4j...")
    
#     with driver.session() as session:
#         count = 0
        
#         for event in events:
#             # 1. Extract Data
#             url = event.get('url', '')
#             title = event.get('title', 'Unknown Event')
#             # Handle date formatting if necessary, GDELT sends YYYYMMDDHHMMSS
#             date_str = event.get('seendate', '20220101') 
            
#             # 2. Heuristic Linking: Find countries mentioned in the title
#             # In a full production system, we would use GDELT's entity extraction features.
#             involved_countries = ['USA'] # Base country from our query
            
#             # Simple keyword check to find other countries
#             if 'China' in title or 'Chinese' in title: involved_countries.append('CHN')
#             if 'Russia' in title or 'Russian' in title: involved_countries.append('RUS')
#             if 'Germany' in title or 'German' in title: involved_countries.append('DEU')
#             if 'Japan' in title or 'Japanese' in title: involved_countries.append('JPN')
#             if 'Mexico' in title or 'Mexican' in title: involved_countries.append('MEX')
#             if 'UK' in title or 'Britain' in title: involved_countries.append('GBR')

#             # 3. Cypher Query: Create Event & Link to Country
#             cypher = """
#             MERGE (e:Event {url: $url})
#             SET e.title = $title, 
#                 e.date = $date,
#                 e.source = 'GDELT'
            
#             # Create relationships for all involved countries
#             WITH e
#             UNWIND $countries AS iso_code
#             MATCH (c:Country {iso_code: iso_code})
#             MERGE (e)-[r:IMPACTS]->(c)
#             SET r.tone = -5.0  # Placeholder tone (real GDELT has specific tone scores)
#             """
            
#             try:
#                 session.run(cypher, {
#                     'url': url, 
#                     'title': title, 
#                     'date': date_str,
#                     'countries': involved_countries
#                 })
#                 count += 1
#             except Exception as e:
#                 print(f"⚠️ Error loading event: {e}")

#     print(f"✅ Successfully loaded {count} events into Neo4j.")

# if __name__ == "__main__":
#     # 1. Fetch
#     events_data = fetch_gdelt_events()
    
#     # 2. Load
#     if events_data:
#         driver = get_driver()
#         if driver:
#             load_events(driver, events_data)
#             close_driver(driver)
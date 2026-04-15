# File: src/db_connection.py
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# 1. Load passwords from the .env file so we don't type them in code
load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def get_driver():
    """Creates a connection to the Neo4j database."""
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        # Verify connectivity
        driver.verify_connectivity()
        print("✅ Successfully connected to Neo4j!")
        return driver
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return None

def close_driver(driver):
    if driver:
        driver.close()

# This part only runs if you run THIS file directly
if __name__ == "__main__":
    driver = get_driver()
    close_driver(driver)
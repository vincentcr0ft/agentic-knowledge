from neo4j import GraphDatabase
import os, sys, traceback

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "neo4j")

print(f"Connecting to {uri} as {user}")
try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        val = session.run("RETURN 1 AS x").single()
        print("Bolt OK:", val["x"]) 
    driver.close()
except Exception as e:
    print("Bolt test failed:", e)
    traceback.print_exc()
    sys.exit(1)

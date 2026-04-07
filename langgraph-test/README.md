Quick LangGraph local test

1. Create venv and install:

   python3 -m venv langgraph-env
   source langgraph-env/bin/activate
   pip install -r requirements.txt

2. Run the minimal test:

   python test_graph.py

3. Optional: start Ollama and Neo4j via Docker (requires Docker):

   docker pull ollama/ollama
   docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

   docker run -d --name neo4j-local -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/test neo4j:latest

Notes: Adjust package list in `requirements.txt` if you prefer a lighter install.

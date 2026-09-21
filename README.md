# Hybrid RAG with Knowledge Graph

A simple Hybrid RAG application that combines **FAISS Vector Search**, **Neo4j Knowledge Graph**, **OpenAI**, **Guardrails**, and **Ragas**.

## Features

- Upload PDF documents
- Extract and split PDF text
- Create FAISS vector embeddings
- Store document entities in Neo4j
- Search using Vector + Knowledge Graph
- Generate answers using OpenAI
- Validate answers with Guardrails
- Evaluate RAG using Ragas
- Streamlit web interface
- Docker deployment
- HTTPS/SSL using Caddy

## Project Structure

```text
New Hybrid RAG/
│
├── Backend/
│   ├── config.py
│   ├── data/
│   ├── evals/
│   ├── guardrails_config.py
│   ├── ingest_pipeline.py
│   ├── llm.py
│   ├── models.py
│   ├── neo4j_kg.py
│   ├── question_pipeline.py
│   └── vector_store/
│
├── frontend/
│   └── app.py
│
├── evals.py
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── Caddyfile
└── .dockerignore
Technologies
Python
LangChain
OpenAI
FAISS
Neo4j
Streamlit
Guardrails AI
Ragas
Docker
Caddy
Setup
1. Create Virtual Environment
python -m venv venv

Activate:

venv\Scripts\activate
2. Install Packages
pip install -r requirements.txt
3. Configure .env
OPENAI_API_KEY=your_openai_api_key

OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j

TOP_K_VECTOR=5
TOP_K_GRAPH=5
Run PDF Ingestion

Place a PDF inside:

Backend/data/

Then run:

python -m Backend.ingest_pipeline

This creates:

PDF
 ↓
Text Chunks
 ↓
FAISS
 ↓
Neo4j Knowledge Graph
Run Application

Start Streamlit:

streamlit run frontend/app.py

Open:

http://localhost:8501

Upload a PDF and ask questions.

Hybrid RAG Flow
User Question
      ↓
FAISS Search
      +
Neo4j Search
      ↓
Combined Context
      ↓
OpenAI LLM
      ↓
Guardrails
      ↓
Final Answer
Run Evaluation
python evals.py

Ragas evaluates:

Faithfulness
Answer Relevancy
Context Precision
Docker

Build the application:

docker compose build

Start containers:

docker compose up -d

Check containers:

docker compose ps

View logs:

docker compose logs -f

Stop containers:

docker compose down
HTTPS

Caddy is used as the HTTPS reverse proxy.

For local testing:

https://localhost

For public deployment, configure your domain in Caddyfile:

rag.example.com {
    reverse_proxy hybrid-rag:8501
}

Caddy can automatically obtain and renew the TLS certificate for a properly configured public domain.

GitHub
git init
git add .
git commit -m "Initial Hybrid RAG project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
Important

Do not upload .env or API keys to GitHub.

Add this to .gitignore:

venv/
.env
__pycache__/
*.pyc
Backend/vector_store/
Project Flow
PDF Upload
    ↓
PDF Processing
    ↓
FAISS + Neo4j
    ↓
Hybrid Retrieval
    ↓
OpenAI
    ↓
Guardrails
    ↓
Answer
    ↓
Ragas Evaluation
Author

Hybrid RAG Project

Built with Python, LangChain, FAISS, Neo4j, OpenAI, Streamlit, Guardrails, Ragas, Docker, and Caddy.
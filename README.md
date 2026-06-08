# Enterprise Multi-Format Retrieval-Augmented Generation (RAG) System

A production-grade, highly optimized Retrieval-Augmented Generation (RAG) application. This repository architecture completely decouples heavy data extraction, mathematical vector processing, and LLM orchestration (**FastAPI** + **FAISS** + **Groq Cloud**) from a conversational, stateful user experience web dashboard (**Streamlit**). 

The entire project workspace environment and runtime dependency matrix are managed by **uv**—the high-performance Python package installer and resolver.

---

## 🏗️ System Architecture & Operational Pipeline

```text
  [ User Interface ]  ---> Adjusts Temperature & System Prompts
          |
          v
   +--------------+
   | Streamlit UI |  ---> Sends Payloads & Streams File Buffers Natively
   +--------------+
          |
          v  HTTP POST (Port 8000)
   +--------------+
   | FastAPI Core |  ---> Leverages App Lifespan Hooks (Models Pre-warmed in RAM)
   +--------------+
      |        |
      |        v [Orchestration Layer]
      |     +-------------------------+
      |     | LangChain + Groq Client | ---> Llama 3.1 Inference Engine
      |     +-------------------------+
      v [Vector Engine]
   +-------------------------+
   | FAISS Index Matrices    | ---> local all-MiniLM-L6-v2 Embeddings
   +-------------------------+
   Rag/
├── src/
│   ├── data_loader.py    # Multi-format document parser (PDF, TXT, CSV, DOCX, XLSX, JSON)
│   ├── embeddings.py     # Local SentenceTransformer embedding neural-net setups
│   ├── vectorstore.py    # Local flat FAISS file read/write serialization index routines
│   └── search.py         # Asynchronous LangChain Groq inference prompt-loop engines
├── main.py               # FastAPI application core, lifespan hooks, and REST gateways
├── frontend.py          # Streamlit conversational chatbot user dashboard portal
├── pyproject.toml       # Declarative PEP 518/621 system framework requirements blueprint
├── uv.lock               # Hard-locked, deterministic dependency tracking lock signature
└── .gitignore            # Secret credential matrices preventing accidental private leaks
Step 1: Configure Your Secret Tokens

Create a .env file in the root workspace folder to safely map your external credentials:
echo "GROQ_API_KEY=gsk_your_actual_groq_cloud_secret_key_here" > .env

Step 2: Spin Up the Server-Side Core (Terminal 1)

Activate the virtual workspace and run the application middleware engine using Uvicorn:
source .venv/bin/activate
uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload

1:Server Verification: Your terminal will output Application startup complete.

2:Developer API Dashboard: Open your browser and head to http://127.0.0.1:8000/docs to view your automatically generated, interactive Swagger UI portal

Step 3: Launch the Streamlit App (Terminal 2)

Open a separate shell terminal, activate your virtual environment, and boot up your user-facing interface:
source .venv/bin/activate
uv run streamlit run frontend.py --server.fileWatcherType none

Note: We include the --server.fileWatcherType none flag to prevent your local Linux kernel from running out of file-system tracking observers (inotify watch limits) when dealing with heavy machine learning virtual folders.

User Screen Link: Your terminal will output a local network URL. Open your browser and navigate straight to http://localhost:8501.
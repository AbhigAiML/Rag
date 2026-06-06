import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Importing your verified modular framework modules
from src.search import RAGSearch

# Load environmental configurations (.env)
load_dotenv()

# Safe, isolated state dictionary for caching singleton assets
rag_application_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context hook. Code before 'yield' runs once when the cloud instance boots.
    This manages resources safely and prevents high-latency reads on incoming API streams.
    """
    # Defensive programming check for Groq routing
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️ WARNING: 'GROQ_API_KEY' is missing from the environment configuration!")

    try:
        print("🔄 Booting Application Context: Instantiating RAG Search Pipelines...")
        
        # This will either load your existing 'faiss_store' or build it automatically from scratch.
        # Overriding the default model string to match a valid Groq production endpoint.
        rag_instance = RAGSearch(llm_model="llama3-8b-8192")
        
        # Cache inside our global context dictionary
        rag_application_state["rag_engine"] = rag_instance
        
        print("🚀 Enterprise RAG Pipeline successfully locked into application memory.")
        yield
    finally:
        # Code here executes when the application container shuts down safely
        rag_application_state.clear()
        print("🛑 Context cleared. Web service instances flushed.")

# Instantiating the core FastAPI server
app = FastAPI(
    title="Faiss-Backed Groq RAG System API",
    description="Production-grade API wrapper for searching and summarizing localized data stores.",
    version="1.0.0",
    lifespan=lifespan
)

# Cross-Origin Resource Sharing (CORS) setup.
# This ensures public frontends, portfolio pages, or test suites can reach your endpoints.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Data Structures for Strict Input/Output Schema Enforcement ---
class QueryPayload(BaseModel):
    query: str = Field(..., min_length=2, example="What is attention mechanism?")
    top_k: int = Field(default=3, description="Number of document chunks to extract from FAISS context matrix.")

class QueryResponse(BaseModel):
    query: str
    answer: str

# --- Operational Endpoints ---
@app.get("/", status_code=status.HTTP_200_OK)
async def service_root():
    """Simple status check node."""
    return {
        "status": "online",
        "engine": "active",
        "interactive_docs_url": "/docs"
    }

@app.post("/api/v1/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def stream_rag_pipeline(payload: QueryPayload):
    """
    Receives incoming queries, extracts matching metadata indices via FAISS, 
    and passes context chunks down into Groq LLM pipelines for inference summary.
    """
    # Safeguard validation checking whether initialization completed smoothly
    if "rag_engine" not in rag_application_state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG search orchestration pipeline is offline or re-indexing."
        )
        
    try:
        # Pull our cached singleton instance
        rag: RAGSearch = rag_application_state["rag_engine"]
        
        # Execute your existing search_and_summarize logic
        generation_output = rag.search_and_summarize(query=payload.query, top_k=payload.top_k)
        
        return QueryResponse(
            query=payload.query,
            answer=str(generation_output)
        )
        
    except Exception as e:
        # Wrap underlying engine errors inside clean internal server exception tracking
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference Engine Processing Bottleneck: {str(e)}"
        )

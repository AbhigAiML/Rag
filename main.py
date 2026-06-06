import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Importing your existing modular framework components
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Thread-safe global storage container to hold our initialized RAG engines
rag_app_context = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager. Anything before 'yield' runs ONCE on server boot.
    This prevents re-initializing FAISS and your LLM on every incoming request.
    """
    try:
        print("🔄 Initializing system memory and loading vector stores...")
        
        # Initialize your FAISS store
        store = FaissVectorStore("faiss_store")
        store.load()
        
        # Initialize your main RAG execution instance
        rag = RAGSearch()
        
        # Cache them securely in our global server context
        rag_app_context["store"] = store
        rag_app_context["rag_engine"] = rag
        
        print("🚀 Enterprise RAG Engine components loaded successfully into memory.")
        yield
    finally:
        # Code here runs when the server shuts down
        rag_app_context.clear()
        print("🛑 Server shutting down. RAG context flushed.")

# Instantiating the web application
app = FastAPI(
    title="Production RAG API Gateway",
    description="Production-grade API layer serving predictions for our FAISS-backed RAG pipeline.",
    version="1.0.0",
    lifespan=lifespan
)

# Cross-Origin Resource Sharing (CORS) middleware configuration.
# This allows any frontend interface or web link to communicate with your backend securely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Request/Response Schema Validation ---
class QueryPayload(BaseModel):
    query: str = Field(..., min_length=3, example="Mining quantitative association rules")
    top_k: int = Field(default=3, description="Number of top documents to retrieve")

class SearchResponse(BaseModel):
    query: str
    result: str

# --- API Operational Endpoints ---
@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """Root health check point to verify code status."""
    return {
        "status": "online",
        "message": "RAG API Framework active. Navigate to /docs for the interactive Swagger dashboard."
    }

@app.post("/api/v1/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def query_rag_pipeline(payload: QueryPayload):
    """
    Accepts user query strings, performs document retrieval via FAISS, 
    and synthesizes summaries via the embedded RAG pipeline engine.
    """
    # Defensive check ensuring the system booted up cleanly
    if "rag_engine" not in rag_app_context:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The underlying RAG engine is currently unavailable or initializing."
        )
        
    try:
        rag: RAGSearch = rag_app_context["rag_engine"]
        
        # Execute the traditional RAG query logic
        summary_output = rag.search_and_summarize(payload.query, top_k=payload.top_k)
        
        return SearchResponse(
            query=payload.query,
            result=str(summary_output)
        )
        
    except Exception as e:
        # Professional standard catch-all logging mechanism safely exposed as HTTP 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution pipeline bottleneck failure: {str(e)}"
        )

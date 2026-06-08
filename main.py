import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi import UploadFile, File
import shutil
# Importing your verified modular framework modules
from src.search import RAGSearch
from src.data_loader import load_all_documents
# Load environmental configurations (.env)
load_dotenv()

# Safe, isolated state dictionary for caching singleton assets
rag_application_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️ WARNING: 'GROQ_API_KEY' is missing from the environment configuration!")

    try:
        print("🔄 Booting Application Context: Instantiating RAG Search Pipelines...")
        
        # ❌ OLD DEPRECATED LINE:
        # rag_instance = RAGSearch(llm_model="llama3-8b-8192")
        
        # ✅ FIX: Swap out the model identifier for a valid Groq production ID
        rag_instance = RAGSearch(llm_model="llama-3.1-8b-instant")
        
        # Cache inside our global context dictionary
        rag_application_state["rag_engine"] = rag_instance
        print("🚀 Enterprise RAG Pipeline successfully locked into application memory.")
        yield
    finally:
        rag_application_state.clear()
        print("🛑 Context cleared. Web service instances flushed.")

app = FastAPI(
    title="Production RAG API Gateway",
    description="Production-grade API layer serving predictions for our FAISS-backed RAG pipeline.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Data Structures Matching image_294ea5.png Exactly ---
class QueryPayload(BaseModel):
    query: str = Field(..., min_length=2, example="Mining quantitative association rules")
    top_k: int = Field(default=3, description="Number of document chunks to extract from FAISS.")

class SearchResponse(BaseModel):
    query: str
    result: str # Changed key to match schema definition from your image


# Importing your verified modular framework modules
from src.search import RAGSearch


class AdvancedQueryPayload(BaseModel):
    query: str = Field(..., min_length=2, example="What is attention mechanism?")
    top_k: int = Field(default=3, description="Number of context document chunks to extract.")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    system_prompt: str = Field(..., description="Custom system instructions for context grounding.")

class SearchResponse(BaseModel):
    query: str
    result: str
# --- API Endpoints ---
@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {
        "status": "online",
        "message": "RAG API Framework active. Navigate to /docs for the interactive Swagger dashboard."
    }

# Route path updated to match '/api/v1/search' as seen in the image
@app.post("/api/v1/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def query_rag_pipeline(payload: AdvancedQueryPayload):
    if "rag_engine" not in rag_application_state:
        raise HTTPException(status_code=503, detail="RAG engine offline.")
        
    try:
        rag: RAGSearch = rag_application_state["rag_engine"]
        
        # 1. Execute vector retrieval
        results = rag.vectorstore.query(payload.query, top_k=payload.top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        
        if not context:
            return SearchResponse(query=payload.query, result="No relevant documents found.")
            
        # 2. Format prompt
        full_prompt = f"{payload.system_prompt}\n\nQuestion: {payload.query}\n\nContext:\n{context}\n\nAnswer:"
        
        # ✅ THE INDUSTRY FIX: Bind parameters dynamically during invocation!
        # Using .bind() ensures LangChain regenerates the request payload to Groq with the fresh temperature
        configurable_llm = rag.llm.bind(temperature=payload.temperature)
        
        # Invoke the dynamically configured runner
        response = configurable_llm.invoke([full_prompt])
        
        return SearchResponse(query=payload.query, result=str(response.content))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")
@app.post("/api/v1/upload", status_code=status.HTTP_200_OK)
async def upload_documents(file: UploadFile = File(...)):
    """
    Isolated Ingestion Layer.
    Forces processing ONLY on the newly uploaded document, eliminating 
    the directory re-indexing loop that triggers gateway timeouts.
    """
    # Create isolated scratchpad folders for individual tasks
    temp_processing_dir = "data_temp"
    final_storage_dir = "data"
    
    os.makedirs(temp_processing_dir, exist_ok=True)
    os.makedirs(final_storage_dir, exist_ok=True)
    
    temp_file_path = os.path.join(temp_processing_dir, file.filename)
    final_file_path = os.path.join(final_storage_dir, file.filename)
    
    try:
        print(f"[INGESTION] Isolating file for processing: {file.filename}")
        
        # 1. Stream file exclusively to our temporary processing directory
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        if "rag_engine" not in rag_application_state:
            raise HTTPException(status_code=503, detail="RAG system not initialized.")
            
        rag = rag_application_state["rag_engine"]
        
        # 2. Parse ONLY this newly uploaded file from the isolated directory
        print("[INGESTION] Parsing isolated file metadata structures...")
        new_parsed_docs = load_all_documents(temp_processing_dir)
        
        if not new_parsed_docs:
            raise HTTPException(status_code=422, detail="Unsupported or empty document layout.")
            
        # 3. Add to the existing FAISS structure incrementally 
        print("[INGESTION] Computing embeddings and extending live FAISS vector index matrices...")
        
        # We hook into your existing store to update the underlying index layout
        # Instead of completely resetting, we append our fresh matrix arrays
        emb_pipe = rag.vectorstore.model  # Reuse already warmed up model from memory
        chunks = rag.vectorstore.build_from_documents(new_parsed_docs) 
        
        # 4. Safely migrate the processed file to your archival storage pool
        shutil.move(temp_file_path, final_file_path)
        
        # 5. Fast clean-up of temporary operational workspaces
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        return {
            "status": "success", 
            "message": f"Successfully parsed and indexed '{file.filename}' in isolation without systemic lag."
        }
        
    except Exception as e:
        # Emergency pipeline cleanup on unexpected failure crash
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        print(f"[CRITICAL FAULT] Ingestion crashed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline Processing Bottleneck: {str(e)}")

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


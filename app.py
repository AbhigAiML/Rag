from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
    
    #docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.load()
    rag=RAGSearch()
    print(rag.search_and_summarize("Mining quantitative association rules", top_k=3))
   

    
import os
import lancedb
import tempfile
from langchain_core.tools import tool
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# --- NEW: Import Reranker ---
from flashrank import Ranker, RerankRequest

# --- Configuration ---
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LANCEDB_URI = "./lancedb_data"
LANCEDB_TABLE = None
EMBEDDINGS = None
RERANKER = None

# --- Helper: Initialize Embeddings ---
def get_embeddings():
    global EMBEDDINGS
    if EMBEDDINGS is None:
        print("--- 🧠 Loading embedding model... ---")
        EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
    return EMBEDDINGS

# --- NEW Helper: Initialize Reranker ---
def get_reranker():
    global RERANKER
    if RERANKER is None:
        print("--- 🧠 Loading Reranker model (FlashRank)... ---")
        # Uses a tiny, ultra-fast model (~4MB)
        RERANKER = Ranker(model_name="ms-marco-TinyBERT-L-2-v2-trt-quantized")
    return RERANKER

def build_rag_index(uploaded_files: list) -> str:
    global LANCEDB_TABLE
    
    if not uploaded_files:
        return "Error: No files uploaded."

    print(f"--- 🧠 Building RAG index from {len(uploaded_files)} files... ---")
    documents = []
    
    # 1. Load Files
    for uploaded_file in uploaded_files:
        file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            if file_ext == ".pdf": loader = PyPDFLoader(tmp_file_path)
            elif file_ext == ".txt": loader = TextLoader(tmp_file_path)
            elif file_ext == ".docx": loader = Docx2txtLoader(tmp_file_path)
            else: continue
            
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading file {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(tmp_file_path): os.remove(tmp_file_path)

    if not documents:
        return "Error: Could not load any valid documents."

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    # 3. Embeddings
    embeddings = get_embeddings()
    doc_data = [{"text": t.page_content, "source": t.metadata.get('source', 'N/A')} for t in texts]
    
    print("--- 🧠 Embedding documents... ---")
    embedded_texts = embeddings.embed_documents([d["text"] for d in doc_data])

    # 4. Save to LanceDB
    try:
        db = lancedb.connect(LANCEDB_URI)
        for i, data in enumerate(doc_data):
            data['vector'] = embedded_texts[i]

        table_name = "documents"
        # Overwrite enables rebuilding the index from scratch
        LANCEDB_TABLE = db.create_table(table_name, data=doc_data, mode="overwrite")
        
        # --- NEW: Create Full-Text Search (FTS) Index ---
        # This enables Hybrid Search (Keyword + Vector)
        print("--- 🧠 Creating Hybrid Search Index (FTS)... ---")
        LANCEDB_TABLE.create_fts_index("text")
        
        return f"Success! Knowledge base built from {len(texts)} chunks. Hybrid Index Ready."

    except Exception as e:
        return f"Error building RAG index: {e}"

@tool
def search_my_documents(query: str) -> str:
    """
    Searches the user's uploaded documents (knowledge base) for an answer.
    Use this tool *first* if the user asks a question about 'my files', 
    'my document', 'the PDF', or a specific topic you know is 
    in their uploaded files.
    """
    global LANCEDB_TABLE
    
    print(f"--- 🧠 [RAG Tool] Searching documents for: '{query}' ---")
    
    # Re-connect if table is lost (e.g. after restart)
    if LANCEDB_TABLE is None:
        try:
            db = lancedb.connect(LANCEDB_URI)
            LANCEDB_TABLE = db.open_table("documents")
        except:
            return "Error: Knowledge base not found. Please upload files first."
    
    try:
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(query)
        
        # --- STEP 1: HYBRID SEARCH ---
        # We fetch MORE results than we need (e.g., 10) to give the Reranker
        # enough options to choose from.
        # query_type="hybrid" uses both Vector Similarity AND Keyword matching.
        initial_results = LANCEDB_TABLE.search(query_vector) \
            .query_type("hybrid") \
            .limit(10) \
            .to_list()
        
        if not initial_results:
            return "No relevant information found in the documents."
        
        # --- STEP 2: RERANKING ---
        # The Reranker looks at the specific query and the text of the results
        # and re-orders them by true relevance.
        print("--- 🧠 Reranking results... ---")
        ranker = get_reranker()
        
        # Convert LanceDB results to FlashRank format
        passages = [
            {"id": str(i), "text": res["text"], "meta": {"source": res["source"]}} 
            for i, res in enumerate(initial_results)
        ]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = ranker.rerank(rerank_request)
        
        # Select Top 3 after reranking
        top_results = reranked_results[:3]
        
        # Format for LLM
        formatted_context = "\n\n--- Context from Documents ---\n"
        for res in top_results:
            source = res['meta'].get('source', 'N/A')
            formatted_context += f"Source ({source}):\n{res['text']}\n-----------------\n"
            
        return formatted_context

    except Exception as e:
        return f"Error during document search: {e}"
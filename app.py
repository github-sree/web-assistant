import os
import pickle
import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain.llms import Ollama
from starlette.middleware.cors import CORSMiddleware

# -----------------------------------------------------------------------------
# Initialize FastAPI application
# -----------------------------------------------------------------------------
app = FastAPI()

# -----------------------------------------------------------------------------
# Enable CORS (Cross-Origin Resource Sharing) so frontend can talk to backend
# -----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (frontend can be hosted anywhere)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Serve static frontend files (index.html, CSS, JS, etc.) from "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------------------------------------------------------
# Directories for storing vector index and knowledge files
# -----------------------------------------------------------------------------
VECTOR_DIR = "vector_store"  # Where FAISS index and embeddings are stored
KB_DIR = "knowledge_base"  # Where uploaded files are stored

# -----------------------------------------------------------------------------
# Load FAISS index and related objects from disk
# -----------------------------------------------------------------------------
# FAISS index stores embeddings for similarity search
index = faiss.read_index(os.path.join(VECTOR_DIR, "faiss_index.bin"))

# "chunks" contain the text splits of your knowledge base documents
chunks = pickle.load(open(os.path.join(VECTOR_DIR, "chunks.pkl"), "rb"))

# SentenceTransformer embedder used for query/document embedding
embedder = pickle.load(open(os.path.join(VECTOR_DIR, "embedder.pkl"), "rb"))

# -----------------------------------------------------------------------------
# Initialize Ollama LLM (self-hosted) for text generation
# -----------------------------------------------------------------------------
# You can replace "tinyllama" with "mistral", "llama3", etc.
llm = Ollama(model="tinyllama")


# -----------------------------------------------------------------------------
# Retrieve top_k most relevant document chunks for a given query
# -----------------------------------------------------------------------------
def retrieve(query, top_k=2):
    """
    Perform semantic search over the FAISS vector store.

    Args:
        query (str): User's question.
        top_k (int): Number of most relevant chunks to retrieve.

    Returns:
        List[str]: List of top_k text chunks most relevant to the query.
    """
    # Convert query into embedding
    q_emb = embedder.encode([query])

    # Perform vector similarity search
    D, I = index.search(np.array(q_emb, dtype="float32"), top_k)

    # Return corresponding text chunks
    return [chunks[i]["page_content"] for i in I[0]]


# -----------------------------------------------------------------------------
# Ask endpoint: Handles Q&A using RAG + Ollama
# -----------------------------------------------------------------------------
@app.post("/ask")
async def ask(query: str = Form(...)):
    """
    API endpoint to process user query and return answer using RAG.

    Steps:
    1. Retrieve relevant chunks from FAISS.
    2. Construct prompt with retrieved context.
    3. Stream answer from Ollama model.
    """
    # Retrieve relevant knowledge
    retrieved = retrieve(query)
    context = "\n".join(retrieved)

    # Strict prompt design to ensure context-based and safe answers
    prompt = f"""
    You are a Healthcare Web Application Assistant.

    Instructions:
    - ONLY use the information provided in the context below to answer.
    - Provide a direct and concise answer to the specific question.
    - Do NOT include information about other modules or topics unless explicitly asked.
    - If the answer is not in the context, reply with exactly: "I don't know."
    - Do NOT summarize the entire system or add unrelated sections.
    - Never generate, infer, or expose Personally Identifiable Information (PII).
    - Use microservices knowledge to answer with the context provided
    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # Inner generator function for streaming output
    def generate():
        """
        Streams tokens/chunks from Ollama model to the client in real-time.
        """
        try:
            for chunk in llm.stream(prompt):  # Stream model output token-by-token
                if chunk:
                    yield chunk
        except Exception as e:
            # If something goes wrong, return error as stream
            yield f"[Error: {str(e)}]"

    # Return response as streaming text
    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8;")


# -----------------------------------------------------------------------------
# Upload endpoint: Allows user to upload new knowledge documents
# -----------------------------------------------------------------------------
@app.post("/upload")
async def upload(file: UploadFile):
    """
    Upload a document (PDF, TXT, CSV, XLSX, etc.) to the knowledge base.

    Note:
        After uploading, you must re-run `rag_pipeline.py` to rebuild FAISS index.
    """
    os.makedirs(KB_DIR, exist_ok=True)
    filepath = os.path.join(KB_DIR, file.filename)

    # Save uploaded file to knowledge_base directory
    with open(filepath, "wb") as f:
        f.write(await file.read())

    return {"status": "File uploaded. Run rag_pipeline.py to rebuild index."}


# -----------------------------------------------------------------------------
# Root endpoint: Serve frontend (index.html)
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    """
    Return frontend index.html file for the web assistant.
    """
    return FileResponse("static/index.html")
import os
import pickle
import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader


# -----------------------------------------------------------------------------
# Load documents from knowledge_base folder
# -----------------------------------------------------------------------------
def load_documents(folder="knowledge_base"):
    """
    Loads documents of various formats (txt, pdf, docx, xlsx, csv) from a folder.

    Args:
        folder (str): Path to the folder containing knowledge base files.

    Returns:
        List[dict]: Each element is a document with 'page_content' and 'metadata'.
    """
    docs = []

    # Iterate through all files in the folder
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        # Handle TXT files
        if file.endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())

        # Handle PDF files
        elif file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())

        # Handle DOCX files
        elif file.endswith(".docx"):
            docs.extend(Docx2txtLoader(path).load())

        # Handle XLSX files (Excel spreadsheets)
        elif file.endswith(".xlsx"):
            df = pd.read_excel(path)
            for _, row in df.iterrows():
                # Convert each row into a text string like "col1: val1 | col2: val2"
                docs.append({
                    "page_content": " | ".join([f"{c}: {row[c]}" for c in df.columns]),
                    "metadata": {"source": file}
                })

        # Handle CSV files
        elif file.endswith(".csv"):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                docs.append({
                    "page_content": " | ".join([f"{c}: {row[c]}" for c in df.columns]),
                    "metadata": {"source": file}
                })

    return docs


# -----------------------------------------------------------------------------
# Split documents into smaller chunks
# -----------------------------------------------------------------------------
def split_documents(docs, chunk_size=800, chunk_overlap=100):
    """
    Splits documents into smaller overlapping text chunks for better embeddings.

    Args:
        docs (List[dict]): List of documents with 'page_content' and 'metadata'.
        chunk_size (int): Maximum characters in each chunk.
        chunk_overlap (int): Overlap between consecutive chunks (to preserve context).

    Returns:
        List[dict]: List of text chunks with 'page_content' and 'metadata'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []

    for d in docs:
        # Handle case if doc is dictionary or LangChain Document object
        text = d["page_content"] if isinstance(d, dict) else d.page_content
        meta = d["metadata"] if isinstance(d, dict) else d.metadata

        # Split text into overlapping chunks
        for c in splitter.split_text(text):
            chunks.append({"page_content": c, "metadata": meta})

    return chunks


# -----------------------------------------------------------------------------
# Build FAISS index from chunks
# -----------------------------------------------------------------------------
def build_faiss(chunks, model="all-MiniLM-L6-v2"):
    """
    Creates a FAISS vector index from text chunks.

    Args:
        chunks (List[dict]): List of text chunks with 'page_content'.
        model (str): HuggingFace SentenceTransformer model for embeddings.

    Returns:
        tuple: (faiss_index, chunks, embedder)
    """
    # Load embedding model
    embedder = SentenceTransformer(model)

    # Convert all chunks into embeddings (vectors)
    embeddings = np.array([embedder.encode(c["page_content"]) for c in chunks], dtype="float32")

    # Get dimension of embeddings
    dim = embeddings.shape[1]

    # Create FAISS index (L2 distance = Euclidean similarity search)
    index = faiss.IndexFlatL2(dim)

    # Add embeddings to FAISS index
    index.add(embeddings)

    return index, chunks, embedder


# -----------------------------------------------------------------------------
# Save FAISS index, chunks, and embedder to disk
# -----------------------------------------------------------------------------
def save_index(index, chunks, embedder, folder="vector_store"):
    """
    Saves FAISS index, chunks, and embedder to disk.

    Args:
        index: FAISS index object.
        chunks: List of text chunks.
        embedder: SentenceTransformer model.
        folder (str): Directory where data is saved.
    """
    os.makedirs(folder, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, os.path.join(folder, "faiss_index.bin"))

    # Save chunks as pickle
    with open(os.path.join(folder, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    # Save embedder as pickle
    with open(os.path.join(folder, "embedder.pkl"), "wb") as f:
        pickle.dump(embedder, f)


# -----------------------------------------------------------------------------
# Main script entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Load documents
    docs = load_documents()

    # Step 2: Split into smaller chunks
    chunks = split_documents(docs)

    # Step 3: Build FAISS index
    index, chunks, embedder = build_faiss(chunks)

    # Step 4: Save FAISS index + supporting data
    save_index(index, chunks, embedder)

    print("FAISS index built and saved!")
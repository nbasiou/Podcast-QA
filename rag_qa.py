import os
import numpy as np
import torch
import faiss

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter



def chunk_transcripts(input_dir, chunk_size=500, overlap=50):
    """
    Splits all transcripts in a directory into smaller retrievable text chunks.

    Parameters:
        input_dir (str): Directory containing transcript files.
        chunk_size (int): Maximum length of each chunk.
        overlap (int): Overlap between chunks to maintain context.

    Returns:
        dict: Dictionary where keys are filenames and values are lists of chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunked_transcripts = {}

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):  # Process only text files
            file_path = os.path.join(input_dir, filename)

            print(f"Processing: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                transcript = f.read()

            chunks = splitter.split_text(transcript)
            print(f"Number of chunks in {filename}: {len(chunks)}")
            print(filename)
            chunked_transcripts[filename] = chunks

    return chunked_transcripts



def store_multiple_transcripts_in_faiss(chunked_transcripts, embedding_model, faiss_index_path):
    """
    Converts multiple transcript chunks into embeddings and stores them in a FAISS vector index.

    Parameters:
        transcripts (dict): Dictionary of filename -> list of text chunks.

    Returns:
        faiss.IndexFlatL2: FAISS index with stored embeddings.
        dict: Mapping of index positions to filenames and chunks.
    """
    all_chunks = []
    index_mapping = {}
    idx = 0

    for filename, chunks in chunked_transcripts.items():
        for chunk in chunks:
            all_chunks.append(chunk)
            index_mapping[idx] = {"filename": filename, "text": chunk}
            idx += 1

    # Create embeddings for all chunks
    doc_embeddings = np.array([embedding_model.encode(chunk) for chunk in all_chunks])

    # Build FAISS index
    dimension = doc_embeddings.shape[1]

    if len(all_chunks) > 10000:  # For large datasets
      faiss_index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW for fast ANN search
    else:
      faiss_index = faiss.IndexFlatL2(dimension)  # Default for small datasets

    faiss_index.add(doc_embeddings)

    # Save FAISS index to disk
    faiss.write_index(faiss_index, faiss_index_path)
    print(f"FAISS index saved at: {faiss_index_path}")

    return faiss_index, index_mapping

def retrieve_relevant_chunks(embedding_model, query, faiss_index, index_mapping, top_k=3):
    """
    Retrieves the most relevant transcript chunks for a given query.

    Parameters:
        query (str): User query.
        faiss_index (faiss.IndexFlatL2): FAISS index containing stored chunks.
        index_mapping (dict): Mapping of FAISS indices to filenames and chunks.
        top_k (int): Number of top results to retrieve.

    Returns:
        list: List of retrieved chunks with filenames.
    """
    query_embedding = np.array([embedding_model.encode(query)])
    distances, indices = faiss_index.search(query_embedding, top_k)

    retrieved_chunks = []
    for i in indices[0]:
        if i in index_mapping:
            retrieved_chunks.append(index_mapping[i])

    return retrieved_chunks


def load_llm_generation_model(model_id):
    """
    Load a large language model for text generation using HuggingFace pipeline.

    Parameters:
        model_id (str): The model identifier from HuggingFace hub (e.g., 'gpt2', 'meta-llama/Llama-2')

    Returns:
        pipe: HuggingFace pipeline object configured for text generation
    """

    pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
    return pipe

def generate_answer(query, context, pipe):
    """
    Generate an AI-powered answer using a loaded LLM generation pipeline.

    Parameters:
        query (str): The user's question.
        context (str): The contextual information or transcript to ground the answer.
        pipe: HuggingFace text-generation pipeline object.

    Returns:
        str: The generated answer from the language model.
    """
    
    prompt = f"Context: {context}\nUser Question: {query}\nAI Answer:"
    response = pipe(prompt, max_length=2000, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]
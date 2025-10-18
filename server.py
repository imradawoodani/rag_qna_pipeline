import logging
import sys
import pickle
import time
import os
from typing import Dict, Any
from functools import lru_cache
import requests
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("MODEL", "llama3.2:3b")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
REQUEST_TIMEOUT = 300
MAX_RETRIES = 3

db = None
text_chunks = None
embedder = None
stats = None
cross_encoder = None
app = Flask(__name__)

def load_index_and_models():
    """Load the FAISS index, chunks, and models into memory."""
    global db, text_chunks, embedder, stats, cross_encoder
    
    try:
        logger.info("Loading text chunks from text_chunks.pkl.")
        with open('text_chunks.pkl', 'rb') as f:
            text_chunks = pickle.load(f)

        logger.info("Loading embedding model: sentence-transformers/all-mpnet-base-v2")
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Loading FAISS index from 'eluvio_index'.")
        db = FAISS.load_local("eluvio_index", embedder, allow_dangerous_deserialization=True)
        try:
            with open('index_stats.pkl', 'rb') as f:
                stats = pickle.load(f)
            logger.info(f"Index stats loaded: {stats.get('total_chunks', 0)} chunks from {stats.get('total_pages', 0)} pages")
        except FileNotFoundError:
            logger.warning("index_stats.pkl not found.")

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {RERANKER_MODEL}")
            cross_encoder = CrossEncoder(RERANKER_MODEL)
            logger.info("Reranker loaded successfully.")
        except Exception as e:
            cross_encoder = None
            logger.warning(f"Reranker model not available, continuing without reranking: {e}")
        
        logger.info("Index and models loaded successfully!")
        return True
        
    except FileNotFoundError:
        logger.critical("Index files (eluvio_index/, text_chunks.pkl) not found.")
        logger.critical("Please run the index.py script first to generate the index.")
        return False
    except Exception as e:
        logger.error(f"Failed to load index: {e}", exc_info=True)
        return False

def preprocess_question(question: str) -> str:
    """Preprocess the question"""
    question = question.strip()
    if len(question.split()) < 3 and "eluvio" not in question.lower():
        question = f"tell me about Eluvio {question}"
    return question

def retrieve_relevant_context(question: str) -> Dict[str, Any]:
    """Retrieve and rerank relevant context"""
    try:
        processed_question = preprocess_question(question)
        docs = db.similarity_search(processed_question, k=20)
        if cross_encoder and docs:
            logger.info(f"Reranking {len(docs)} documents.")
            pairs = [(processed_question, d.page_content) for d in docs]
            scores = cross_encoder.predict(pairs)
            doc_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            docs = [doc for doc, score in doc_scores][:RERANK_TOP_K]
            logger.info("Reranking complete.")
        else:
            docs = docs[:RERANK_TOP_K]
        context_parts = []
        sources = set()
        for doc in docs:
            metadata = doc.metadata or {}
            source_url = metadata.get('url', 'Unknown')
            sources.add(source_url)
            context_parts.append(f"Source URL: {source_url}\nContent: {doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)
        return {
            'context': context,
            'sources': list(sources),
            'processed_question': processed_question
        }
        
    except Exception as e:
        logger.error(f"Error retrieving context: {e}", exc_info=True)
        return {'context': "", 'sources': [], 'processed_question': question}

def create_prompt(question: str, context: str) -> str:
    """Creates the final prompt for the LLM."""
    return f"""You are an expert representative for Eluvio. Your tone is professional, confident, and direct.

Strictly adhere to these rules:
1. Your entire response must be based *only* on the provided context below.
2. If the answer is not in the context, you must reply with the single sentence: "I'm sorry, I do not have knowledge about this." Do not add any other words.
3. Do not mention the context, sources, or that you are an AI. Do not use phrases like "Based on the provided information".
4. Be specific. Use product and technology names from the context. Avoid vague marketing language.

Here is an example question and answer:
Question: What does Eluvio do?
Answer: Eluvio provides a blockchain-based platform called the Content Fabric for distributing and monetizing digital content, particularly video. It replaces traditional media clouds and CDNs by offering a global, decentralized network that handles live and file-based content publishing, transcoding, streaming, and minting digital collectibles like NFTs.

Given the following context below, answer the question {question}. Strictly adhere to the rules mentioned.

Context:

{context}

Answer:"""

def call_ollama(prompt: str) -> str:
    """Calls the Ollama API"""
    payload = {"model": MODEL, "prompt": prompt, "stream": False, "options": {
            "num_predict": 1024
        }}
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            answer = response.json().get("response", "").strip()
            if answer:
                return answer
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(2 ** attempt)
    return "I apologize, but I'm currently unable to process your request. Please try again later."

def ask_agent(question: str) -> Dict[str, Any]:
    """Generation pipeline."""
    start_time = time.time()
    context_data = retrieve_relevant_context(question)

    if not context_data.get('context'):
        answer = "I'm sorry, I do not have knowledge about this."
    else:
        prompt = create_prompt(question, context_data['context'])
        answer = call_ollama(prompt)
    
    processing_time = time.time() - start_time
    logger.info(f"Question processed in {processing_time:.2f}s. Answer: '{answer[:100]}...'")
    return {
        "response": answer,
        "metadata": {
            "processing_time": round(processing_time, 2),
            "sources_used": context_data.get('sources', []),
            "model_used": MODEL,
            "question_processed": context_data.get('processed_question', question)
        }
    }

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or not (question := data.get("prompt", "").strip()):
        return jsonify({"error": "Missing or empty 'prompt' field"}), 400
    if len(question) > 1000:
        return jsonify({"error": "Prompt is too long (max 1000 chars)"}), 400
    
    result = ask_agent(question)
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    ollama_status = "disconnected"
    try:
        if requests.get(OLLAMA_URL.replace("/api/generate", "/"), timeout=3).status_code == 200:
            ollama_status = "connected"
    except requests.exceptions.RequestException:
        pass

    return jsonify({
        "status": "healthy" if db and ollama_status == "connected" else "degraded",
        "index_loaded": bool(db),
        "reranker_available": bool(cross_encoder),
        "chunks_available": len(text_chunks) if text_chunks else 0,
        "ollama_status": ollama_status,
        "model": MODEL,
    })

if __name__ == "__main__":
    logger.info("Starting Eluvio RAG Server!")
    if not load_index_and_models():
        sys.exit(1)
    logger.info("Server ready! Listening on http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
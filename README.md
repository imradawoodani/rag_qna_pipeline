# Eluvio Q/A RAG Challenge

## Overview

This project implements a complete Retrieval-Augmented Generation system to answer questions about Eluvio. It features a robust data pipeline that scrapes public content from `eluv.io` using **Playwright** to handle modern JavaScript and anti-bot measures. The scraped content is then processed into a local **FAISS** vector index and served via a Flask API that uses a local LLM through **Ollama**.

## Key Features

- **Robust Web Scraping**: Uses **Playwright** to control a real browser engine, reliably rendering JavaScript and evading common anti-bot systems that block simpler scrapers.
- **High-Quality Embeddings**: Leverages `sentence-transformers/all-mpnet-base-v2` for dense, 768-dimensional embeddings that enable nuanced semantic search.
- **Smart Chunking**: Optimized chunking with overlap and metadata preservation.
- **Prompt Engineering**: Employs carefully engineered prompts to enforce a professional, brand-consistent voice, with strict guardrails to prevent hallucination and reject generic "marketing-speak" answers.
- **Retrieve-and-Rerank Pipeline**: Implements a two-stage retrieval process. First, a broad vector search retrieves a set of candidate documents, which are then re-ordered for precision by a `CrossEncoder` reranking model.

## Setup

Follow these steps to set up the environment and run the full RAG pipeline.

* PREREQUISITES:
  * **Conda**: You must have Conda installed to manage the environment.
  * **Ollama**: You must have Ollama installed and running on your machine.

* STEP 1: CREATE AND ACTIVATE THE CONDA ENVIRONMENT
  This project is configured to run on **Python 3.12**. These commands will create a dedicated Conda environment and activate it.

```bash
# Create a new conda environment named "rag_env" with Python 3.12
conda create --name rag_env python=3.12

# Activate the environment
conda activate rag_env
```
* STEP 2: INSTALL DEPENDENCIES
  Install the required Python packages and the browser engine for Playwright.

```bash
# Install Python packages from the requirements file
pip install -r requirements.txt

# Install the Chromium browser for Playwright
playwright install
```

* STEP 3: PREPARE THE LLM
  Pull the required model using Ollama. The application is configured for llama3.2:3b by default, but you can use a more powerful model like llama3:8b for higher-quality answers.

```bash
ollama pull llama3.2:3b
(Ensure the Ollama application is running in the background).
```

* STEP 4: BUILD THE VECTOR INDEX
  Run the indexing script. This will launch a browser window to scrape the Eluvio website and build the FAISS index.

```bash
python index.py
Note: A browser window will open and navigate through the pages automatically. Let it run until the script reports "Index creation complete!".
```

* STEP 5: START THE API SERVER
  Once the index is built, start the Flask server.

```bash
python server.py
The server will start on http://localhost:5001.
```

* STEP 6: QUERY THE API
You can now send requests to the server to ask questions.

```bash
curl -X POST "http://localhost:5001/ask" \
     -H "Content-Type: application/json" \
     -d '{"prompt":"What is the Eluvio Content Fabric?"}'
```

## PROJECT STRUCTURE

- index.py: The data pipeline script. It crawls eluv.io, scrapes content using Playwright, and builds the FAISS vector index.
- server.py: The Flask API server. It loads the index files and provides endpoints to ask questions.
- requirements.txt: A list of all required Python packages, with pinned versions for stability.

## Strategy & Implementation Details

### WEB SCRAPING WITH PLAYWRIGHT
**Problem:** Simple requests-based scraping failed to extract content, and requests-html was blocked or crashed.

**Solution:** I used Playwright, a modern browser automation library. The index.py script launches a full Chromium browser, emulates a real user profile (user agent, screen size), and intelligently waits for the page's network activity to idle before extracting the fully rendered HTML. This approach is highly resilient to antibot systems.

### RETRIEVAL: SIMILARITY SEARCH + RERANKING
To ensure the most relevant context is provided to the LLM:

**Broad Retrieval:** The system first performs a fast vector similarity_search to retrieve the top 20 potentially relevant document chunks.

**Fine-Grained Reranking:** These 20 candidates are then passed to a CrossEncoder model (cross-encoder/ms-marco-MiniLM-L-6-v2). This more powerful model scores the relevance of each chunk to the query, and the system selects the top 5 from this reranked list.

### PROMPT ENGINEERING & GUARDRAILS
The quality of the final answer depends heavily on the instructions given to the LLM.

**Persona:** The prompt instructs the model to speak as an official Eluvio representative.

**Grounding:** The model is strictly forbidden from using any information outside the provided context.

**Refusal:** If the answer cannot be found in the context, the model is clearly told they are allowed to reply with the exact phrase: I'm sorry, I do not have knowledge about this. This attempts to prevent hallucinations.

**Sanitization:** The server code automatically removes phrases like "Based on the provided context..." if the model mistakenly adds them.


## Updated Examples

Below are example interactions that reflect the brand voice and behaviors enforced by the system.

### Q: "What does Eluvio do?"

```json
{
  "metadata":{
    "model_used":"llama3.2:3b",
    "processing_time":15.94,
    "question_processed":"What does Eluvio do?",
    "sources_used":[
      "https://eluv.io/careers",
      "https://eluv.io/monetization/media-wallet",
      "https://eluv.io/content-fabric/technology",
      "https://eluv.io/content-fabric",
      "https://eluv.io/av-core/fabric-core"
    ]},
    "response":"Eluvio provides a blockchain-based platform called the Content Fabric for distributing and monetizing digital content, particularly video, by harnessing breakthroughs in machine learning, blockchain security, advanced cryptography, and low cost compute to achieve efficiency and simplicity."
}
```

### Q: "What technology does Eluvio use?"

```json
{
  "metadata":{
    "model_used":"llama3.2:3b",
    "processing_time":13.16,
    "question_processed":"What technology does Eluvio use",
    "sources_used":[
      "https://eluv.io/content-fabric",
      "https://eluv.io/careers",
      "https://eluv.io/content-fabric/technology"
    ]},
    "response":"Eluvio uses a combination of Ethereum and Polkadot Substrate forks as the basis for its active Fabric networks for production and test, with full stack implementations built on both blockchains."}
```

## API Endpoints

### POST /ask
Ask a question about Eluvio.

**Request**:
```json
{
  "prompt": "What does Eluvio do?"
}
```

**Response**:
```json
{
  "response": "Detailed answer...",
  "metadata": {
    "processing_time": 45.2,
    "sources_used": 5,
    "model_used": "llama3.2:3b",
    "sources": [...]
  }
}
```

### GET /health
Check system health and status.

**Response**:
```json
{
  "status": "healthy",
  "index_loaded": true,
  "chunks_available": 6,
  "ollama_status": "connected",
  "model": "llama3.2:3b",
  "stats": {
    "total_chunks": 6,
    "total_pages": 3,
    "avg_chunk_size": 625
  }
}
```

## Artifacts

Running `index.py` produces the following artifacts in the project root:
- `eluvio_index/` – FAISS index data directory
- `text_chunks.pkl` – serialized list of enhanced chunks with metadata
- `index_stats.pkl` – summary statistics about the built index
- `raw_pages.jsonl` – debug dump of scraped pages (URL, title, content)

## Future Enhancements

1. **Multi-Modal Support**: Image and video content processing
2. **Real-Time Updates**: Live content synchronization
3. **Advanced Analytics**: User behavior and query analytics
4. **A/B Testing**: Response quality optimization
5. **Scalability**: Distributed processing and load balancing
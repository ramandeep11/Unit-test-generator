# RAG (Retrieval-Augmented Generation) System

A Flask-based RAG system with OCR capabilities that processes text and images from webpages and PDFs to answer user queries using local LLMs via Ollama.

## Overview

This project implements a comprehensive RAG pipeline that combines:
- Document processing (PDF and web content)
- Image text extraction using Tesseract OCR
- Vector storage with Chroma
- LLM-powered question answering using Qwen 2.5 Coder
- Interactive Gradio UI for seamless user interaction

## Architecture

```
┌─────────────┐
│   User      │
│   Input     │
└──────┬──────┘
       │
       ▼
┌──────────────────┐         ┌─────────────────┐
│  Gradio UI       │────────▶│  Flask API      │
│  (ChatUI.py)     │         │  (rag.py)       │
└──────────────────┘         └────────┬────────┘
                                      │
                     ┌────────────────┼────────────────┐
                     │                │                │
                     ▼                ▼                ▼
              ┌─────────────┐  ┌──────────┐  ┌──────────────┐
              │  Tesseract  │  │  Chroma  │  │   Ollama     │
              │     OCR     │  │  Vector  │  │ (Qwen 2.5)   │
              └─────────────┘  │   Store  │  └──────────────┘
                               └──────────┘
```

## Core Components

### 1. rag.py - Flask API Server

The main backend service providing multiple endpoints for document processing and querying.

#### Configuration

```python
embed = FastEmbedEmbeddings()  # Embedding model
text_split = RecursiveCharacterTextSplitter(
    chunk_size=2054,
    chunk_overlap=250
)
llm = Ollama(model="qwen2.5-coder:7b")  # Local LLM
vector_DB = "VDB2"  # Vector database directory
```

#### API Endpoints

##### `/llm` (POST)
**Purpose**: Direct LLM queries without RAG

**Input**:
```json
{
  "query": "Your question here"
}
```

**Output**: Plain text response from LLM

**Use Case**: Simple queries that don't require document context

---

##### `/pdf` (POST)
**Purpose**: Upload and process PDF documents into vector store

**Input**: Multipart form data with file upload

**Process**:
1. Save PDF to `pdf/` directory
2. Load and split PDF using PDFPlumberLoader
3. Create embeddings and store in Chroma
4. Persist vector store

**Output**: Confirmation message with file path

---

##### `/ask_pdf` (POST)
**Purpose**: Query against previously processed PDFs

**Input**:
```json
{
  "query": "Question about PDF content"
}
```

**Process**:
1. Load existing vector store (VDB2)
2. Similarity search with threshold (k=20, score_threshold=0.1)
3. Retrieve relevant chunks
4. Generate answer using LLM with context

**Output**:
```json
{
  "answer": "Generated answer",
  "sources": [
    {
      "source": "document.pdf",
      "page_content": "relevant chunk"
    }
  ]
}
```

---

##### `/extract_text` (POST)
**Purpose**: Extract text from images on a webpage

**Input**:
```json
{
  "url": "https://example.com"
}
```

**Process**:
1. Fetch webpage HTML
2. Parse with BeautifulSoup
3. Find all image tags
4. Extract text from images using Tesseract OCR
5. Handle SVG images separately (extract XML text)

**Output**:
```json
{
  "extracted_text": "Combined text from all images"
}
```

---

##### `/tesseract` (POST) ⭐ Primary Endpoint
**Purpose**: Full RAG pipeline with webpage text and image OCR

**Input**:
```json
{
  "url": "https://example.com or /path/to/local/file.html",
  "query": "Your question"
}
```

**Process**:
1. **Fetch Content**: Support both web URLs and local file paths
2. **Parse HTML**: Extract text using BeautifulSoup
3. **Image Processing**:
   - Find all `<img>` tags
   - Download images
   - Preprocess images:
     - Convert to grayscale
     - Enhance contrast (2x)
     - Apply binary threshold
     - Median filter for noise reduction
   - SVG handling: Convert to PNG using cairosvg
   - Extract text using Tesseract OCR
4. **Combine**: Merge webpage text and OCR results
5. **Vectorize**: Split into chunks and store in temporary Chroma DB
6. **Retrieve**: Similarity search for relevant chunks
7. **Generate**: LLM generates answer with context
8. **Cleanup**: Delete temporary vector store

**Output**:
```json
{
  "answer": "Contextual answer from webpage and images",
  "sources": [
    {
      "source": "url",
      "page_content": "relevant chunk"
    }
  ]
}
```

**Special Features**:
- Comprehensive image preprocessing for better OCR accuracy
- Supports both raster (PNG, JPG) and vector (SVG) images
- Temporary vector store (deleted after query)
- Uses specialized prompt for webpage Q&A

---

##### `/summarize` (POST)
**Purpose**: Summarize webpage with advanced image processing (experimental)

**Input**:
```json
{
  "url": "https://example.com",
  "query": "Summarization request"
}
```

**Process**:
- Similar to `/tesseract` but uses `process_image()` function
- Implements LayoutLM and graph-based OCR (advanced features)
- Currently uses PaddleOCR and PyTorch geometric graphs

**Note**: This endpoint includes experimental features for document understanding using LayoutLMv2

---

##### `/fusion_rag` (POST)
**Purpose**: Placeholder for future FusionRAG implementation

**Status**: Not fully implemented (returns placeholder response)

---

##### `/local` (POST)
**Purpose**: Query local HTML files (hardcoded path)

**Input**:
```json
{
  "path": "file_path",
  "question": "your question"
}
```

**Note**: Currently hardcoded to read from specific path, needs generalization

---

### 2. ChatUI.py - Gradio Interface

Interactive web UI for the RAG system.

**Features**:
- URL input field for webpage/document source
- Query input for user questions
- Chatbot interface displaying conversation history
- Follow-up question capability
- Maintains conversation context across queries

**Components**:
1. **Main Input Section**:
   - URL textbox
   - Query textbox
   - Submit button

2. **Conversation Display**:
   - Chatbot component showing Q&A history

3. **Follow-up Section**:
   - Follow-up question textbox
   - Follow-up submit button

**Backend Connection**: Makes POST requests to `http://localhost:8080/tesseract`

---

### 3. embedding-models.py

Testing script for comparing different embedding models.

**Purpose**: Evaluate and compare embedding model performance

**Models Tested**:
- `all-MiniLM-L6-v2` (lightweight, fast)
- `all-mpnet-base-v2` (balanced)
- `multi-qa-mpnet-base-dot-v1` (Q&A optimized)
- `paraphrase-multilingual-mpnet-base-v2` (multilingual)

**Process**:
1. Load sample chunks about ML topics
2. Generate embeddings for chunks and query
3. Calculate cosine similarity scores
4. Rank chunks by relevance
5. Compare results across models

**Use Case**: Determine which embedding model works best for your use case

---

### 4. live_ollama.py

Demonstration of streaming LLM responses.

**Purpose**: Show real-time token streaming from Ollama

**Features**:
- Custom callback handler for streaming
- Real-time token printing
- Uses Llama 3.1 model

**Use Case**: Example implementation for streaming responses in production

---

## Image Processing Pipeline

The system uses advanced image preprocessing for optimal OCR results:

```
Original Image
     │
     ▼
Grayscale Conversion
     │
     ▼
Contrast Enhancement (2x)
     │
     ▼
Binary Thresholding (128)
     │
     ▼
Median Filter (3x3)
     │
     ▼
Tesseract OCR
     │
     ▼
Extracted Text
```

## Prompts

### PDF/Document Prompt
```
You are a technical assistant good at searching documents.
If you do not have an answer from the provided information say so.
Try to give answer in 250 words or less.
```

### Webpage Prompt
```
You are a knowledgeable assistant tasked with answering questions
based on the content of a webpage. Guidelines:
1. Answer using only the information provided in context
2. State clearly if context is insufficient
3. Avoid assumptions or adding external information
4. Provide concise but complete answers
5. Base opinions on facts from context only
6. State inability to answer if question is unrelated to context
```

## Installation

### Prerequisites

```bash
# Install Ollama
# Visit https://ollama.ai for installation instructions

# Pull required model
ollama pull qwen2.5-coder:7b

# Install Tesseract OCR
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### Python Dependencies

```bash
cd RAG

# Install required packages
pip install flask flask-cors
pip install langchain langchain-community
pip install chromadb fastembed
pip install pypdf pdfplumber
pip install beautifulsoup4 requests
pip install pytesseract pillow cairosvg
pip install paddleocr transformers torch torch-geometric
pip install gradio sentence-transformers scikit-learn
```

## Usage

### Start Flask API Server

```bash
python rag.py
# Server runs on http://0.0.0.0:8080
```

### Launch Gradio UI

```bash
python ChatUI.py
# UI opens in browser automatically
```

### Example API Calls

#### Upload PDF
```bash
curl -X POST http://localhost:8080/pdf \
  -F "file=@document.pdf"
```

#### Query PDF
```bash
curl -X POST http://localhost:8080/ask_pdf \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?"}'
```

#### Query Webpage with OCR
```bash
curl -X POST http://localhost:8080/tesseract \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "query": "Summarize this page"}'
```

### Test Embedding Models

```bash
python embedding-models.py
# Outputs similarity scores for different models
```

### Test Streaming

```bash
python live_ollama.py
# Demonstrates real-time token streaming
```

## Directory Structure

```
RAG/
├── rag.py                    # Main Flask API server
├── ChatUI.py                 # Gradio web interface
├── embedding-models.py       # Embedding model comparison
├── live_ollama.py           # Streaming example
├── pdf/                     # PDF storage directory
├── VDB2/                    # Persistent vector database
└── README.md                # This file
```

## Technical Details

### Vector Database
- **Engine**: Chroma
- **Embeddings**: FastEmbed
- **Chunk Size**: 2054 characters
- **Chunk Overlap**: 250 characters
- **Similarity Search**: k=20, threshold=0.1

### OCR Configuration
- **Engine**: Tesseract
- **Image Preprocessing**:
  - Grayscale conversion
  - 2x contrast enhancement
  - Binary thresholding (threshold=128)
  - 3x3 median filter for noise reduction
- **SVG Handling**: cairosvg for rasterization

### LLM Configuration
- **Model**: Qwen 2.5 Coder 7B (via Ollama)
- **Temperature**: Default
- **Context Window**: Model dependent
- **Streaming**: Supported (see live_ollama.py)

## Features

✅ PDF document processing and querying
✅ Webpage content extraction and analysis
✅ Image text extraction with Tesseract OCR
✅ SVG image support
✅ Local and remote file support
✅ Conversation history in UI
✅ Source attribution in responses
✅ Temporary vector stores for one-off queries
✅ Advanced image preprocessing for better OCR
✅ Multi-model embedding comparison tools
✅ Streaming response support

## Limitations

- `/summarize` endpoint uses experimental features (LayoutLMv2) that may require additional setup
- `/local` endpoint has hardcoded file path
- `/fusion_rag` is not fully implemented
- Tesseract OCR accuracy depends on image quality
- Large documents may exceed LLM context window
- Vector store (VDB2) grows with PDF uploads

## Future Enhancements

- [ ] Implement FusionRAG endpoint fully
- [ ] Add support for more document formats (DOCX, TXT, etc.)
- [ ] Optimize chunk size and overlap dynamically
- [ ] Add caching for frequently accessed documents
- [ ] Implement authentication and rate limiting
- [ ] Support for multiple languages in OCR
- [ ] Batch processing for multiple documents
- [ ] Advanced document layout understanding with LayoutLMv2
- [ ] Graph neural networks for document structure analysis

## Troubleshooting

### Tesseract Not Found
```bash
# Ensure tesseract is in PATH
tesseract --version

# If not found, install and add to PATH
export PATH="/usr/local/bin:$PATH"
```

### Ollama Connection Error
```bash
# Verify Ollama is running
ollama list

# Start Ollama service if needed
ollama serve
```

### Vector Store Errors
```bash
# Clear vector database
rm -rf VDB2/

# Vector store will be recreated on next PDF upload
```

## Performance Tips

1. **For large PDFs**: Increase chunk size to reduce number of chunks
2. **For better accuracy**: Preprocess images before OCR
3. **For faster queries**: Use smaller embedding models
4. **For memory efficiency**: Use temporary vector stores when possible
5. **For production**: Implement response caching and connection pooling
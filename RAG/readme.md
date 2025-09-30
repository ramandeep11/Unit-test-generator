# RAG Local Project

This project implements a Retrieval-Augmented Generation (RAG) system with multiple endpoints and a user-friendly Gradio interface. The system can process both text and images from web pages or local files.

## Features

- Text extraction and processing from web pages
- Image text extraction using Tesseract OCR
- Interactive chat interface using Gradio

## API Endpoints

### 1. /llm (POST)
- Handles direct LLM queries using the Qwen 2.5 Coder model
- Input: JSON with "query" field
- Output: Text response from the LLM

### 2. /ask_pdf (POST)
- Processes PDF documents using vector store retrieval
- Uses Chroma vector database for document storage
- Implements similarity search with threshold
- Input: JSON with "query" field
- Output: Contextual response based on PDF content

### 4. /tesseract (POST)
- Dedicated endpoint for image processing
- Extracts text from images using Tesseract OCR
- Handles both web-hosted and local images
- Input: JSON with "url" and "query" fields
- Output: Processed text from images

## Gradio User Interface

The project includes a user-friendly chat interface built with Gradio that provides:

- URL input field for webpage/document source
- Query input for user questions
- Conversation history display
- Follow-up question capability
- Real-time response viewing

### UI Components:
1. Main input section:
   - URL input field
   - Query input field
   - Submit button

2. Chat interface:
   - Conversation history display
   - Follow-up question input
   - Follow-up submit button

The UI maintains conversation context and provides a seamless experience for document-based Q&A interactions.

## Technical Stack

- Flask for API endpoints
- Gradio for user interface
- Tesseract OCR for image processing
- LayoutLMv2 for document understanding (not used in this version)
- Chroma for vector storage
- Qwen 2.5 Coder as the base LLM

# Generative AI Projects

A collection of three independent AI-powered automation and information retrieval projects leveraging LLMs, RAG, and multi-agent systems.

## Projects

### 1. [Unit-test-generator](./Unit-test-generator/)

**Type**: Multi-Agent Automation System

**Description**: An intelligent unit test generation system built with CrewAI that automatically generates, writes, and validates unit tests for Java projects. The system uses multiple specialized AI agents organized into crews that work together to clone repositories, generate comprehensive unit tests, validate them through builds, and commit results to a new branch.

**Key Features**:
- Automated unit test generation for Java service classes
- Iterative refinement based on build feedback (up to 3 attempts)
- CrewAI Flow orchestration with multiple specialized crews
- Git integration for cloning, branching, and committing
- Maven/Gradle build validation
- Custom tools for file operations and project building

**Tech Stack**: CrewAI, OpenAI GPT-4.1-mini, Python, Git, Maven/Gradle

**[→ View Full Documentation](./Unit-test-generator/README.md)**

---

### 2. [RAG](./RAG/)

**Type**: Retrieval-Augmented Generation System

**Description**: A Flask-based RAG system with OCR capabilities that processes text and images from webpages and PDFs to answer user queries. The system combines document processing, image text extraction using Tesseract OCR, vector storage with Chroma, and LLM-powered question answering using local models via Ollama.

**Key Features**:
- PDF document processing and querying
- Webpage content extraction with image OCR support
- Advanced image preprocessing for accurate text extraction
- Vector storage with Chroma and FastEmbed embeddings
- Interactive Gradio UI with conversation history
- Multiple API endpoints for different use cases
- Source attribution in responses

**Tech Stack**: Flask, LangChain, Ollama (Qwen 2.5 Coder), Tesseract OCR, Chroma, Gradio, BeautifulSoup

**[→ View Full Documentation](./RAG/README.md)**

---

### 3. [LLM_with_Web](./LLM_with_Web/)

**Type**: LLM-Powered Web Search & Analysis

**Description**: An intelligent campaign information retrieval system that combines LLM capabilities with web search and scraping to extract current marketing campaign details for companies across different regions. The system searches the web using DuckDuckGo, scrapes relevant content, and uses AI to generate concise campaign summaries.

**Key Features**:
- DuckDuckGo web search integration
- Automated web scraping with BeautifulSoup
- LLM-powered content analysis and summarization
- Multi-region support (Japan, US, UK)
- Simple Gradio UI for easy interaction
- Single-line campaign summaries
- Local LLM execution (no API keys required)

**Tech Stack**: DuckDuckGo Search API, BeautifulSoup, LangChain, Ollama (Deepseek-R1), Gradio

**[→ View Full Documentation](./LLM_with_Web/README.md)**

---

## Common Prerequisites

All projects require:

### Ollama
```bash
# Install Ollama for local LLM inference
# Visit https://ollama.ai for installation

# Pull required models
ollama pull qwen2.5-coder:7b    # For RAG project
ollama pull deepseek-r1         # For LLM_with_Web project
```

### Python
- Python 3.10+ (Unit-test-generator requires <3.14)
- Virtual environment recommended

### Environment Setup
```bash
# For Unit-test-generator
export OPENAI_API_KEY="your-api-key"

# For RAG (install Tesseract)
brew install tesseract  # macOS
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
```

---

## Quick Start

### Unit-test-generator
```bash
cd Unit-test-generator
pip install -e .
kickoff
```

### RAG
```bash
cd RAG
pip install flask langchain chromadb gradio pytesseract beautifulsoup4
python rag.py  # Start API server
python ChatUI.py  # Launch UI (in separate terminal)
```

### LLM_with_Web
```bash
cd LLM_with_Web
pip install gradio duckduckgo-search beautifulsoup4 langchain
python ScrapeUI.py
```

---

## Project Comparison

| Feature | Unit-test-generator | RAG | LLM_with_Web |
|---------|-------------------|-----|--------------|
| **Primary Use** | Test automation | Document Q&A | Web research |
| **LLM Provider** | OpenAI API | Ollama (Local) | Ollama (Local) |
| **Framework** | CrewAI | Flask + LangChain | LangChain |
| **UI** | CLI | Gradio | Gradio |
| **Agents** | Multi-agent (4 crews) | Single pipeline | Single agent |
| **Data Source** | Git repositories | PDFs, Webpages | Web search |
| **Output** | Generated code | Answers + sources | Campaign summaries |

---

## Architecture Patterns

### Unit-test-generator: Multi-Agent Flow
```
Start → GitClone → GenerateTest (iterative) → GitCommit → End
```

### RAG: Pipeline Pattern
```
Document → Split → Embed → VectorDB → Retrieve → LLM → Answer
```

### LLM_with_Web: Search-Analyze Pattern
```
Query → Search → Scrape → LLM Analysis → Summary
```

---

## License

See individual project directories for license information.

---

## Contributing

Each project is independent. Please refer to individual project READMEs for specific contribution guidelines.

---

## Support

For issues or questions:
1. Check the individual project README
2. Review troubleshooting sections
3. Ensure all prerequisites are installed
4. Verify Ollama is running (for RAG and LLM_with_Web)
5. Check API key configuration (for Unit-test-generator)
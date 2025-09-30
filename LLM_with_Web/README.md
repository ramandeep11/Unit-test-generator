# LLM with Web Search

An intelligent campaign information retrieval system that combines LLM capabilities with web search and scraping to extract current marketing campaign details for companies across different regions.

## Overview

This project leverages DuckDuckGo search API, BeautifulSoup web scraping, and Deepseek-R1 LLM to automatically discover and summarize the latest marketing campaigns for any company. The system searches the web, scrapes relevant content, and uses AI to extract concise campaign information.

## Architecture

```
┌─────────────────┐
│   Gradio UI     │
│  (ScrapeUI.py)  │
└────────┬────────┘
         │
         │ company_name, region
         ▼
┌─────────────────────────────┐
│    Agent Class              │
│    (agent.py)               │
├─────────────────────────────┤
│  ┌─────────────────────┐   │
│  │  DuckDuckGo Search  │   │
│  └──────────┬──────────┘   │
│             │               │
│             ▼               │
│  ┌─────────────────────┐   │
│  │ BeautifulSoup       │   │
│  │ Web Scraper         │   │
│  └──────────┬──────────┘   │
│             │               │
│             ▼               │
│  ┌─────────────────────┐   │
│  │  Ollama LLM         │   │
│  │  (Deepseek-R1)      │   │
│  └─────────────────────┘   │
└─────────────────────────────┘
         │
         ▼
    Campaign Summary
```

## Components

### 1. agent.py - Core Agent System

The main logic that orchestrates search, scraping, and LLM processing.

#### Agent Class

**Purpose**: Handles DuckDuckGo web searches

**Methods**:
- `search(query, region)`: Performs DuckDuckGo search with regional filtering
  - **Parameters**:
    - `query` (str): Search query string
    - `region` (str): Region code (e.g., "jp-jp", "us-en", "uk-en")
  - **Returns**: List of search results (max 1 result)
  - **Result Structure**:
    - `href`: URL of the result
    - `body`: Snippet/description

- `example_search(company_name, region)`: Demo method showing search usage
  - Searches for: "What are the current campaigns for {company_name}"
  - Prints result details

**Initialization**:
```python
agent = Agent()
results = agent.search("campaign query", "us-en")
```

---

#### scrape_website Class

**Purpose**: Complete pipeline for searching, scraping, and analyzing campaign information

**Attributes**:
- `agent`: Instance of Agent class for searches
- `prompt`: LangChain PromptTemplate for LLM processing

**LLM Configuration**:
```python
llm = Ollama(model="deepseek-r1")
```

**Prompt Template**:
```
You are a helpful assistant that can scrape a website and return the content.
Here is the content of the website:
{content}
Return single line from each website related to the campaigns for {company_name}
Do not return any other text than the single line.
And try to keep it short and concise.
```

**Methods**:

##### `scrape(url)`
Extracts clean text from a webpage.

**Parameters**:
- `url` (str): URL to scrape

**Process**:
1. Fetch webpage using requests
2. Parse HTML with BeautifulSoup
3. Remove script and style tags
4. Extract text content
5. Clean and format text

**Returns**: Cleaned text string

**Example**:
```python
scraper = scrape_website()
content = scraper.scrape("https://example.com")
```

---

##### `get_compaign_from_website_content(company_name, region)`
Main method that orchestrates the entire workflow.

**Parameters**:
- `company_name` (str): Name of the company to search for
- `region` (str): Region code for search localization

**Process**:
1. **Search**: Query DuckDuckGo for current campaigns
   - Query: "What are the current campaigns for {company_name}"
2. **Scrape**: For each search result:
   - Fetch webpage content
   - Clean and extract text
3. **Analyze**: For each scraped page:
   - Feed content to LLM with prompt
   - Extract single-line campaign summary
4. **Combine**: Join all results with newlines

**Returns**: String with campaign summaries (one per line)

**Example**:
```python
scraper = scrape_website()
campaigns = scraper.get_compaign_from_website_content("Nike", "us-en")
print(campaigns)
# Output: "Nike launches 'Just Do It' 2024 campaign..."
```

**Flow Diagram**:
```
Input: company_name, region
         │
         ▼
    DuckDuckGo Search
    "What are the current campaigns for {company}"
         │
         ▼
    Get top result URLs
         │
         ▼
    For each URL:
         │
         ├─▶ Scrape webpage
         │   Extract text
         │
         ├─▶ Send to LLM
         │   Extract campaign info
         │
         └─▶ Collect result
         │
         ▼
    Join all results
         │
         ▼
    Return campaign summary
```

---

### 2. ScrapeUI.py - Gradio Interface

Interactive web UI for campaign information retrieval.

**Function**: `create_ui()`

Creates and returns a Gradio Blocks interface.

**UI Components**:

1. **Input Row**:
   - **Company Name Textbox**: Enter company to search for
   - **Region Dropdown**: Select region
     - Options: "jp-jp" (Japan), "us-en" (USA), "uk-en" (UK)
     - Default: "us-en"

2. **Output Section**:
   - **Campaign Results Textbox**: Displays extracted campaigns (5 lines)

3. **Submit Button**: Triggers the search and scraping process

**Handler Function**: `get_campaigns(company, region)`

**Process**:
1. Instantiate `scrape_website` class
2. Call `get_compaign_from_website_content()`
3. Return results or error message

**Error Handling**:
- Try-except block catches all exceptions
- Returns formatted error message on failure

**Example Usage**:
```python
# User enters "Coca-Cola" and selects "us-en"
# System returns: "Coca-Cola's 'Real Magic' campaign focuses on..."
```

---

## Workflow

### Complete End-to-End Flow

```
1. User Input
   ├─ Company: "Starbucks"
   └─ Region: "us-en"
        │
        ▼
2. Search Phase (Agent.search)
   └─ DuckDuckGo Query: "What are the current campaigns for Starbucks"
        │
        ▼
3. Result Retrieval
   └─ Top 1 result URL retrieved
        │
        ▼
4. Scraping Phase (scrape_website.scrape)
   ├─ Fetch webpage HTML
   ├─ Remove scripts/styles
   └─ Extract clean text
        │
        ▼
5. LLM Analysis Phase (LangChain + Deepseek)
   ├─ Feed webpage content to LLM
   ├─ Apply campaign extraction prompt
   └─ Get single-line summary
        │
        ▼
6. Output
   └─ Display: "Starbucks launches 'Every Name's A Story' personalization campaign..."
```

---

## Installation

### Prerequisites

```bash
# Install Ollama
# Visit https://ollama.ai

# Pull Deepseek-R1 model
ollama pull deepseek-r1
```

### Python Dependencies

```bash
cd LLM_with_Web

# Install required packages
pip install gradio
pip install duckduckgo-search
pip install beautifulsoup4
pip install requests
pip install langchain
pip install langchain-community
```

---

## Usage

### Start the Application

```bash
python ScrapeUI.py
# Gradio interface launches automatically in browser
```

### Using the Interface

1. **Enter Company Name**: Type the company you want to search for
   - Example: "Apple", "Microsoft", "Tesla"

2. **Select Region**: Choose the search region
   - `jp-jp`: Japan (Japanese results)
   - `us-en`: United States (English results)
   - `uk-en`: United Kingdom (English results)

3. **Click "Get Campaigns"**: System will:
   - Search the web
   - Scrape relevant pages
   - Extract campaign information
   - Display results

### Example Queries

```
Company: Nike
Region: us-en
Result: "Nike's 'Move to Zero' sustainability campaign aims for zero carbon..."

Company: McDonald's
Region: uk-en
Result: "McDonald's UK launches 'Good to Know' transparency campaign..."

Company: Toyota
Region: jp-jp
Result: "トヨタの「未来をつくる」キャンペーンが開始..."
```

---

## Configuration

### Changing LLM Model

Edit `agent.py`:

```python
# Current
llm = Ollama(model="deepseek-r1")

# Alternative models
llm = Ollama(model="llama3.1")
llm = Ollama(model="mistral")
llm = Ollama(model="qwen2.5")
```

### Adjusting Search Results

Edit `agent.py`:

```python
# Current: Returns 1 result
def search(self, query, region):
    return self.ddgs.text(query, region, max_results=1)

# Change to get more results
return self.ddgs.text(query, region, max_results=5)
```

### Customizing Prompt

Edit the prompt in `agent.py` `scrape_website.__init__()`:

```python
self.prompt = PromptTemplate(
    template="""
    Your custom instructions here...
    Content: {content}
    Company: {company_name}
    """,
    input_variables=["content", "company_name"]
)
```

### Adding More Regions

Edit `ScrapeUI.py`:

```python
region_input = gr.Dropdown(
    choices=[
        "jp-jp",  # Japan
        "us-en",  # USA
        "uk-en",  # UK
        "de-de",  # Germany
        "fr-fr",  # France
        "ca-en",  # Canada
    ],
    label="Region",
    value="us-en"
)
```

---

## Technical Details

### Search Configuration
- **Engine**: DuckDuckGo Search API
- **Max Results**: 1 per query
- **Regional Support**: Multiple regions supported
- **Rate Limiting**: None (DuckDuckGo's limits apply)

### Scraping Configuration
- **Parser**: BeautifulSoup (html.parser)
- **Content Cleaning**: Removes scripts, styles
- **Text Extraction**: Full page text content

### LLM Configuration
- **Model**: Deepseek-R1 (via Ollama)
- **Framework**: LangChain
- **Chain Type**: LLMChain
- **Temperature**: Default (not specified)
- **Output**: Single-line campaign summary

---

## Features

✅ Web search with regional filtering
✅ Automatic web scraping
✅ LLM-powered content analysis
✅ Multi-region support (Japan, US, UK)
✅ Clean, single-line campaign summaries
✅ Simple Gradio UI
✅ Error handling and reporting
✅ Local LLM execution (no API keys)

---

## Limitations

- Only retrieves top 1 search result per query
- Depends on DuckDuckGo search availability
- Scraping may fail on JavaScript-heavy sites
- LLM output quality depends on webpage content
- No authentication or rate limiting
- No result caching
- No historical campaign tracking

---

## Future Enhancements

- [ ] Increase search results for better coverage
- [ ] Add JavaScript rendering for dynamic sites
- [ ] Implement result caching
- [ ] Support for more regions
- [ ] Historical campaign tracking
- [ ] Export results to CSV/JSON
- [ ] Add source URL attribution
- [ ] Implement retry logic for failed scrapes
- [ ] Add sentiment analysis of campaigns
- [ ] Support for competitor comparison

---

## Troubleshooting

### Ollama Model Not Found
```bash
# Check installed models
ollama list

# Pull required model
ollama pull deepseek-r1

# Verify Ollama is running
ollama serve
```

### DuckDuckGo Search Fails
```bash
# Check internet connection
ping duckduckgo.com

# Update duckduckgo-search package
pip install --upgrade duckduckgo-search
```

### Gradio Won't Launch
```bash
# Check if port 7860 is in use
lsof -i :7860

# Use different port
# Edit ScrapeUI.py: demo.launch(server_port=7861)
```

### Empty Results
- Check if company name is spelled correctly
- Try different regions
- Check if Ollama service is running
- Verify internet connectivity

---

## Performance Tips

1. **Faster searches**: Use more specific company names
2. **Better results**: Try multiple regions for global companies
3. **Reliability**: Ensure stable internet connection
4. **LLM speed**: Use smaller/faster Ollama models for quick responses
5. **Resource usage**: Close unused Ollama models to free memory

---

## Directory Structure

```
LLM_with_Web/
├── agent.py          # Core search, scrape, and LLM logic
├── ScrapeUI.py       # Gradio web interface
├── LICENSE           # License file
└── README.md         # This file
```

---

## Use Cases

- **Marketing Research**: Discover competitor campaigns
- **Brand Monitoring**: Track your company's campaign presence
- **Market Analysis**: Compare campaigns across regions
- **Content Research**: Find campaign messaging and themes
- **Competitive Intelligence**: Understand market positioning
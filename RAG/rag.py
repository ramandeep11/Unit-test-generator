from flask import Flask , request, jsonify
from flask_cors import CORS
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import urllib.request
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen

from PIL import Image
import io
import pytesseract
from urllib.parse import urljoin
from paddleocr import PaddleOCR
from transformers import LayoutLMv2Processor, LayoutLMv2ForSequenceClassification
import torch
import torch_geometric
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
from PIL import ImageFilter, ImageEnhance
import cairosvg

embed = FastEmbedEmbeddings()

text_split = RecursiveCharacterTextSplitter(
    chunk_size= 2054, chunk_overlap=250, length_function= len, is_separator_regex=False
)

raw_prompt2 = PromptTemplate.from_template(
        """ 
        <s>[INST] You are a knowledgeable assistant tasked with answering questions based on the content of a webpage. Your role is to provide accurate and relevant information from the given context. Please follow these guidelines:

        1. Answer the question using only the information provided in the context.
        2. If the context doesn't contain enough information to answer the question fully, state that clearly.
        3. Avoid making assumptions or adding information not present in the context.
        4. Provide concise answers, but ensure they are complete and accurate.
        5. If asked for an opinion or judgment, base it solely on the facts presented in the context.
        6. If the question is not related to the context at all, politely state that you can't answer based on the given information.

        Remember, your goal is to be helpful while maintaining accuracy and truthfulness based on the webpage content provided. [/INST] </s>
        [INST] Question: {input}
        Context from webpage: {context}
        Answer:
        [/INST]
        """
    )




def extract_text_from_image(image_url ):
        try:
            response = requests.get(image_url)
            img = Image.open(io.BytesIO(response.content))
            return pytesseract.image_to_string(img)
        except Exception as e:
            print(f"Error processing image {image_url}: {str(e)}")
            return ""
        

def extract_text_and_layout(img_url):
    # Download image
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    
    # OCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(img, cls=True)
    
    # Extract text and bounding boxes
    words = []
    boxes = []
    for line in result:
        for word_info in line:
            words.append(word_info[1][0])
            boxes.append(word_info[0])
    
    # LayoutLM processing
    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
    encoding = processor(img, words, boxes=boxes, return_tensors="pt")
    
    # LayoutLM model (for feature extraction)
    model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased")
    with torch.no_grad():
        outputs = model(**encoding)
    
    # Extract features
    features = outputs.hidden_states[-1]  # Use the last hidden state
    
    return words, boxes, features

def build_graph(words, boxes, features):
    # Create nodes
    x = features.squeeze(0)  # Node features
    
    # Create edges (connect nearby words)
    edges = []
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            if abs(boxes[i][0] - boxes[j][0]) < 50 or abs(boxes[i][1] - boxes[j][1]) < 20:
                edges.append([i, j])
                edges.append([j, i])  # Add both directions
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create graph
    data = torch_geometric.data.Data(x=x, edge_index=edge_index)
    
    return data

def process_image(img_url):
    words, boxes, features = extract_text_and_layout(img_url)
    graph = build_graph(words, boxes, features)
    
    # Here you would typically run your GNN model on the graph
    # For simplicity, we'll just return the words and their connections
    connected_words = []
    for edge in graph.edge_index.t():
        connected_words.append(f"{words[edge[0]]} - {words[edge[1]]}")
    
    return " ".join(words), connected_words


raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. Try to give answer in 250 words or less. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

vector_DB = "VDB2"

llm = Ollama(model="qwen2.5-coder:7b")

app = Flask(__name__)
CORS(app)

@app.route('/llm',methods=["POST"])
def llmPost():
    print("inside the post function")
    json= request.json
    query= json.get("query")
    
    print(query)
    
    response= llm.invoke(query)
    
    return "Answer:"+response
    


@app.route('/ask_pdf',methods=["POST"])
def ask_Pdf():
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=vector_DB, embedding_function=embed)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    document_chain = create_stuff_documents_chain(llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    print(result)

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer

@app.route("/pdf",methods=["POST"])
def pdfPost():
    file =  request.files["file"]
    file_name = file.filename
    save_file = "pdf/"+ file_name
    file.save(save_file)
    # print(save_file)
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    
    chunks = text_split.split_documents(docs)
    
    vector_store = Chroma.from_documents(documents=chunks, embedding=embed, persist_directory= vector_DB)
    
    vector_store.persist()
    
    return "saved file to path: "+ save_file+" fileName: "+file_name
   
@app.route("/extract_text", methods=["POST"])
def extract_text_endpoint():
    json_content = request.json
    url = json_content.get("url")

    # Fetch webpage content
    response = requests.get(url)
    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch webpage"}), 400

    # Parse content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get images from the webpage
    img_tags = soup.find_all('img')
    
    extracted_texts = []

    for img in img_tags:
        src = img.get('src')
        if src:
            img_url = urljoin(url, src)
            try:
                img_response = requests.get(img_url, stream=True)
                img_response.raise_for_status()

                content_type = img_response.headers.get('Content-Type', '')
                
                if 'svg' in content_type or src.lower().endswith('.svg'):
                    # Process SVG directly
                    svg_content = img_response.content.decode('utf-8')
                    svg_soup = BeautifulSoup(svg_content, 'xml')
                    svg_text = svg_soup.get_text(strip=True)
                    if svg_text:
                        extracted_texts.append(svg_text)
                elif 'image' in content_type:
                    # Use tesseract for non-SVG images (png, jpeg, etc.)
                    image_text = extract_text_from_image(img_url)
                    if image_text:
                        extracted_texts.append(image_text)
            except Exception as e:
                print(f"Error processing image {img_url}: {str(e)}")

    # Combine all extracted texts
    combined_text = ' '.join(extracted_texts)

    return jsonify({"extracted_text": combined_text})

@app.route("/fusion_rag", methods=["POST"])
def fusion_rag_endpoint():
    json_content = request.json
    url = json_content.get("url")
    query = json_content.get("query")

    # Fetch webpage content or read from local file
    if url.startswith('http://') or url.startswith('https://'):
        response = requests.get(url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch webpage"}), 400
        content = response.text
    else:
        try:
            with open(url, 'r', encoding='utf-8') as file:
                content = file.read()
        except FileNotFoundError:
            return jsonify({"error": "Local file not found"}), 400
        except IOError:
            return jsonify({"error": "Error reading local file"}), 400

    # Parse and clean content
    soup = BeautifulSoup(content, 'html.parser')
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)

    # Extract text from images
    img_tags = soup.find_all('img')
    image_texts = []

    for img in img_tags:
        src = img.get('src')
        if src:
            img_url = urljoin(url, src)
            try:
                image_text = extract_text_from_image(img_url)
                if image_text:
                    image_texts.append(image_text)
            except Exception as e:
                print(f"Error processing image {img_url}: {str(e)}")

    # Combine all texts
    combined_text = cleaned_text + ' ' + ' '.join(image_texts)

    # Implement FusionRAG logic here
    # This is a placeholder for the actual FusionRAG implementation
    # You would need to replace this with your specific FusionRAG logic
    fusion_rag_result = implement_fusion_rag(combined_text, query)

    return jsonify({"answer": fusion_rag_result})

def implement_fusion_rag(text, query):
    # Placeholder for FusionRAG implementation
    # Replace this with your actual FusionRAG logic
    # This might involve:
    # 1. Splitting the text into chunks
    # 2. Embedding the chunks and the query
    # 3. Retrieving relevant chunks
    # 4. Fusing the retrieved information
    # 5. Generating a response using a language model
    
    # For now, we'll just return a placeholder response
    return f"FusionRAG processed query: {query} on the given text."


@app.route("/tesseract", methods=["POST"])
def tesseract_endpoint():
    json_content = request.json
    url = json_content.get("url")
    query = json_content.get("query")

    # Fetch webpage content or read from local file
    if url.startswith('http://') or url.startswith('https://'):
        response = requests.get(url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch webpage"}), 400
        content = response.text
    else:
        try:
            with open(url, 'r', encoding='utf-8') as file:
                content = file.read()
        except FileNotFoundError:
            return jsonify({"error": "Local file not found"}), 400
        except IOError:
            return jsonify({"error": "Error reading local file"}), 400

    # Parse and clean content
    soup = BeautifulSoup(content, 'html.parser')
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)

    # Get images from the webpage
    img_tags = soup.find_all('img')
    image_texts = []

    for img in img_tags:
        src = img.get('src')
        if src:
            img_url = urljoin(url, src)
            print("img_url test69")
            print(img_url)
            try:
                img_response = requests.get(img_url, stream=True)
                img_response.raise_for_status()

                content_type = img_response.headers.get('Content-Type', '')
                
                # Check if the content type is an image or SVG
                if 'image' not in content_type:
                    continue

                # Process SVG images by converting to PNG
                if 'svg' in content_type:
                    try:
                        svg_data = img_response.content
                        png_data = cairosvg.svg2png(bytestring=svg_data)
                        image = Image.open(io.BytesIO(png_data))
                    except Exception as e:
                        print(f"Error processing SVG image: {e}")
                        continue
                else:
                    # Process raster images directly from the response content
                    try:
                        image = Image.open(io.BytesIO(img_response.content))
                    except UnidentifiedImageError:
                        print("Error: Unable to open image")
                        continue

                # Preprocess the image
                image = image.convert('L')  # Convert to grayscale

                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2.0)
                
                image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Convert to binary

                # Apply noise reduction
                image = image.filter(ImageFilter.MedianFilter(size=3))

                # Perform OCR
                ocr_text = pytesseract.image_to_string(image)
                print("ocr_text test70")
                print(ocr_text)

                alt_text = img.get('alt', '')
                if ocr_text:
                    image_texts.append(f"Image with text: {alt_text}. OCR extracted text: {ocr_text}")

            except requests.RequestException as e:
                print(f"Request error: {e}")
                continue

    # Combine webpage text and image texts
    combined_text = cleaned_text + "\n\n" + "\n\n".join(image_texts)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents(
    [combined_text],
    [{"source": url}] * len(combined_text)  # Adding "source" metadata to each document
)

    # Create and persist vector store
    vector_store = Chroma.from_documents(documents=docs, embedding=embed, persist_directory=vector_DB)
    vector_store.persist()

    # Create retriever and chain
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    document_chain = create_stuff_documents_chain(llm, raw_prompt2)
    chain = create_retrieval_chain(retriever, document_chain)

    # Run query
    result = chain.invoke({"input": query})

    # Prepare response
    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}

    # Clean up the vector store
    vector_store.delete_collection()
    vector_store.persist()

    return jsonify(response_answer)



@app.route("/summarize", methods=["POST"])
def summarize_webpage():
    json_content = request.json
    url = json_content.get("url")
    query = json_content.get("query")

    # Fetch webpage content
    response = requests.get(url)
    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch webpage"}), 400

    # Parse and clean content
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    print(text)
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)


    # Get images from the webpage
    img_tags = soup.find_all('img')
    image_texts = []

    for img in img_tags:
        src = img.get('src')
        if src:
            # Make sure we have absolute URL
            img_url = urljoin(url, src)
            alt_text = img.get('alt', '')
            ocr_text, connected_words = process_image(img_url)
            if ocr_text:
                image_texts.append(f"Image with text: {alt_text}. OCR extracted text: {ocr_text}")
                image_texts.append(f"Connected words in the image: {', '.join(connected_words)}")

    # Combine webpage text and image texts
    combined_text = cleaned_text + "\n\n" + "\n\n".join(image_texts)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([combined_text])

    # Create and persist vector store
    vector_store = Chroma.from_documents(documents=docs, embedding=embed, persist_directory=vector_DB)
    vector_store.persist()

    # Create retriever and chain
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )
    raw_prompt2 = PromptTemplate.from_template(
        """ 
        <s>[INST] You are a knowledgeable assistant tasked with answering questions based on the content of a webpage. Your role is to provide accurate and relevant information from the given context. Please follow these guidelines:

        1. Answer the question using only the information provided in the context.
        2. If the context doesn't contain enough information to answer the question fully, state that clearly.
        3. Avoid making assumptions or adding information not present in the context.
        4. Provide concise answers, but ensure they are complete and accurate.
        5. If asked for an opinion or judgment, base it solely on the facts presented in the context.
        6. If the question is not related to the context at all, politely state that you can't answer based on the given information.

        Remember, your goal is to be helpful while maintaining accuracy and truthfulness based on the webpage content provided. [/INST] </s>
        [INST] Question: {input}
        Context from webpage: {context}
        Answer:
        [/INST]
        """
    )

    document_chain = create_stuff_documents_chain(llm, raw_prompt2)
    chain = create_retrieval_chain(retriever, document_chain)


    # Invoke the chain with the query
    result = chain.invoke({"input": query})

    # Prepare the response
    sources = []
    for doc in result["context"]:
        sources.append({
            
            "source": doc.metadata.get("source", "Webpage"),
            "page_content": doc.page_content
        })

    response_answer = {"answer": result["answer"], "sources": sources}
    return jsonify(response_answer)

    

@app.route("/local",methods=["POST"])
def AskAI():
    json_content = request.json
    url = json_content.get("path")
    query = json_content.get("question")
    
    webpage_content = open("/Users/ramandeep.maan/Desktop/llama3/llama3-quant/RAG/test.html",'r').read()
    soup = BeautifulSoup(webpage_content, 'html.parser')
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    raw_prompt2 = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching docuemnts and I will provide you the content of a article. Answer the questions based on the information provided and  If you do not have an answer from the provided information say so. Try to answer in 5 words or less. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
    """
    )
    
    llm_chain = LLMChain(llm=llm, prompt=raw_prompt2)

    result = llm_chain.invoke({"input": query,"context": text})
    
    return result
    
  
def load_and_split_text(text, chunk_size=1000):
    words = text.split()
    chunks = []
    chunk = []
    chunk_length = 0

    for word in words:
        if chunk_length + len(word) + 1 > chunk_size:
            chunks.append(' '.join(chunk))
            chunk = []
            chunk_length = 0
        chunk.append(word)
        chunk_length += len(word) + 1  # +1 for the space

    if chunk:
        chunks.append(' '.join(chunk))

    return chunks

    
def start_app():
    app.run(host="0.0.0.0",port=8080, debug=True)

if __name__ == "__main__":
    start_app()


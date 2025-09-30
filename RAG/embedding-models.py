# try new embedding model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


models = [
    'all-MiniLM-L6-v2',
    'all-mpnet-base-v2', 
    'multi-qa-mpnet-base-dot-v1',
    'paraphrase-multilingual-mpnet-base-v2'
]

chunks = [
    "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses statistical techniques to give computer systems the ability to progressively improve performance on a specific task.",
    "Deep learning uses neural networks with multiple layers to progressively extract higher level features from raw input. For example, in image processing, lower layers may identify edges, while higher layers may identify concepts relevant to a human such as digits or letters or faces.",
    "Natural language processing deals with the interaction between computers and human language, particularly how to program computers to process and analyze large amounts of natural language data. It combines computational linguistics, machine learning, and deep learning.",
    "Computer vision focuses on image recognition tasks and enables computers to understand and process visual information from the world. It involves methods for acquiring, processing, analyzing, and understanding digital images to produce numerical or symbolic information.",
    "Reinforcement learning involves agents learning through actions and rewards in an environment. The agent learns to make decisions by performing actions and receiving feedback in the form of rewards or penalties, similar to how humans learn through trial and error.",
    "Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second task. It's particularly popular in deep learning because it allows training deep neural networks with comparatively little data.",
    "Supervised learning is a type of machine learning where the model learns from labeled training data. The algorithm learns to map input features to known output labels, making it useful for classification and regression tasks.",
    "Unsupervised learning algorithms work with unlabeled data to discover hidden patterns or groupings. Common applications include clustering, dimensionality reduction, and anomaly detection in datasets.",
    "Generative AI models can create new content including text, images, music, and more. These models learn patterns from training data and can generate new, similar content that wasn't explicitly programmed.",
    "AutoML (Automated Machine Learning) aims to automate the process of applying machine learning to real-world problems, including tasks like feature selection, algorithm selection, and hyperparameter optimization."
]


query = "What is machine learning?"

def get_similarity_scores(model_name, texts, query):
    model = SentenceTransformer(model_name)
    
    
    chunk_embeddings = model.encode(texts)
    query_embedding = model.encode([query])[0]
    
   
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
    return similarities


for model_name in models:
    print(f"\nTesting model: {model_name}")
    print("-" * 50)
    
    try:
        similarities = get_similarity_scores(model_name, chunks, query)
        
        chunk_scores = list(zip(chunks, similarities))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        chunk_scores = [((' '.join(chunk.split()[:10]) + '...'), score) for chunk, score in chunk_scores]
        for chunk, score in chunk_scores:
            print(f"Score: {score:.4f} | Chunk: {chunk}")
            
    except Exception as e:
        print(f"Error with model {model_name}: {str(e)}")

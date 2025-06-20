# cloud_config.py
# Configuration for cloud deployment (without local Ollama)

import os

# Cloud deployment settings
IS_CLOUD_DEPLOYMENT = os.environ.get('RENDER') or os.environ.get('HEROKU') or os.environ.get('STREAMLIT_CLOUD')

if IS_CLOUD_DEPLOYMENT:
    # Use Hugging Face API for cloud deployment
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'  # Hugging Face model
    LANGUAGE_MODEL = 'huggingface'  # Use HF Inference API
    
    # API settings
    HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY', '')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    
    # Use external APIs
    USE_HUGGINGFACE_API = True
    USE_LOCAL_OLLAMA = False
    
    print("☁️ Cloud deployment mode activated")
    print(f"🔗 Using HuggingFace embeddings: {EMBEDDING_MODEL}")
    
else:
    # Local development with Ollama
    from config import *
    USE_HUGGINGFACE_API = False
    USE_LOCAL_OLLAMA = True
    
    print("🏠 Local development mode")
    print(f"🔗 Using Ollama: {LANGUAGE_MODEL}")

# RAG parameters (same for both)
TOP_K_RETRIEVAL = 3
CHUNK_SIZE = 200
SIMILARITY_THRESHOLD = 0.5

GENERATION_OPTIONS = {
    'temperature': 0.2,
    'top_p': 0.8,
    'max_length': 100,
    'stop': ['\n\n', 'Question:', 'Context:']
}
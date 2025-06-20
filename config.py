# config.py - Updated with working models

# Model configurations - WORKING SETUP
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'  # ✅ Working perfectly
LANGUAGE_MODEL = 'tinyllama:1.1b'  # ✅ Fixed - was 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Ollama configuration
OLLAMA_URL = 'http://localhost:11434'
USE_HTTP_API = True  # Required to bypass Python library issues

# RAG parameters
TOP_K_RETRIEVAL = 3
CHUNK_SIZE = 200  # characters per chunk
SIMILARITY_THRESHOLD = 0.5

# Generation options optimized for tinyllama:1.1b
GENERATION_OPTIONS = {
    'temperature': 0.2,      # Lower for more factual responses
    'top_p': 0.8,           # Nucleus sampling
    'num_predict': 100,     # Max tokens to generate
    'stop': ['\n\n', 'Question:', 'Context:']  # Stop tokens
}

# Vector database (in-memory for this demo)
VECTOR_DB = []

# System settings
SUPPRESS_WARNINGS = True  # Suppress SSL and other warnings

print("✅ Config loaded with working models:")
print(f"   🔗 Embedding: {EMBEDDING_MODEL}")
print(f"   🤖 Language: {LANGUAGE_MODEL}")
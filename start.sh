#!/bin/bash
# start.sh - Startup script for deployment

echo "🚀 Starting RAG Application..."

# Start Ollama in background
echo "📡 Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to be ready..."
for i in {1..30}; do
  if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama is ready!"
    break
  fi
  echo "⏳ Waiting... ($i/30)"
  sleep 2
done

# Pull required models
echo "📦 Pulling required models..."
ollama pull tinyllama:1.1b
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf

# Create data file if it doesn't exist
if [ ! -f "cat-facts.txt" ]; then
  echo "📚 Creating cat facts data..."
  python3 download_data.py
fi

# Start Streamlit
echo "🎨 Starting Streamlit app..."
streamlit run streamlit_app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false

# Cleanup on exit
trap "kill $OLLAMA_PID" EXIT
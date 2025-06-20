# 🤖 RAG Chat Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions about cats using your own documents!

## 🎯 Features

- 📚 **Document Processing**: Automatically chunks and processes text documents
- 🔍 **Semantic Search**: Find relevant information using vector embeddings
- 🤖 **AI Responses**: Generate contextual answers using language models
- 💬 **Chat Interface**: Beautiful Streamlit web interface
- 🚀 **Easy Deployment**: Deploy to Render, Heroku, or Streamlit Cloud

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-chat-assistant.git
   cd rag-chat-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama (for local development)**
   ```bash
   ollama serve
   ollama pull tinyllama:1.1b
   ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
   ```

4. **Run the application**
   ```bash
   python download_data.py  # Download cat facts
   streamlit run streamlit_app.py
   ```

### Cloud Deployment

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## 📋 Project Structure

```
rag-chat-assistant/
├── src/                    # RAG system modules
│   ├── preprocessor.py     # Document preprocessing
│   ├── embeddings.py       # Embedding generation
│   ├── vector_database.py  # Vector storage
│   └── retrieval_system.py # Main RAG system
├── streamlit_app.py        # Web interface
├── config.py               # Configuration
├── download_data.py        # Data setup
├── requirements.txt        # Dependencies
├── Dockerfile             # Container setup
└── README.md              # This file
```

## 🛠️ Configuration

### Local Setup (config.py)
```python
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'tinyllama:1.1b'
OLLAMA_URL = 'http://localhost:11434'
TOP_K_RETRIEVAL = 3
CHUNK_SIZE = 200
```

### Cloud Setup
Set environment variables:
- `HUGGINGFACE_API_KEY`: For embeddings
- `OPENAI_API_KEY`: For language generation (optional)

## 🎮 Usage

1. **Ask Questions**: Type questions about cats in the chat interface
2. **View Sources**: See which document chunks were used for the answer
3. **Adjust Settings**: Use the sidebar to modify search parameters
4. **Clear Chat**: Start fresh conversations anytime

### Example Questions
- "How fast can cats run?"
- "How much do cats sleep?"
- "What can you tell me about cat hearing?"

## 🧪 RAG System Components

### 1. Document Preprocessing
- Text cleaning and normalization
- Intelligent chunking strategies
- Metadata extraction

### 2. Embedding Generation
- Vector embeddings for semantic search
- Batch processing for efficiency
- Caching for performance

### 3. Vector Database
- In-memory storage with persistence
- Fast similarity search
- Metadata filtering

### 4. Retrieval System
- Query processing
- Context ranking
- Response generation

## 📊 Performance

- **Response Time**: ~2-3 seconds for typical queries
- **Accuracy**: High relevance using semantic search
- **Scalability**: Handles hundreds of documents efficiently

## 🛡️ Deployment Options

### Render (Recommended)
1. Connect your GitHub repository
2. Set environment variables if using external APIs
3. Deploy with one click

### Streamlit Cloud
1. Connect GitHub repository
2. Set `streamlit_app.py` as main file
3. Add secrets for API keys

### Docker
```bash
docker build -t rag-chat .
docker run -p 8501:8501 rag-chat
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Ollama**: Local language model inference
- **Streamlit**: Web interface framework
- **Hugging Face**: Embedding models
- **Render**: Cloud deployment platform

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/rag-chat-assistant/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/rag-chat-assistant/discussions)
- 📧 **Email**: your.email@example.com

---

**Made with ❤️ for the AI community**

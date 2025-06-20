# src/retrieval_system.py
# Step 5: Complete Retrieval System Implementation

import requests
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import warnings
import time
import sys
import os

# Add current directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
    print("✅ Using existing config.py")
except ImportError:
    print("⚠️ config.py not found, using default settings")
    # Fallback config
    class config:
        EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
        LANGUAGE_MODEL = 'tinyllama:1.1b'
        OLLAMA_URL = 'http://localhost:11434'
        TOP_K_RETRIEVAL = 3
        CHUNK_SIZE = 200
        GENERATION_OPTIONS = {
            'temperature': 0.2,
            'top_p': 0.8,
            'num_predict': 100,
            'stop': ['\n\n', 'Question:', 'Context:']
        }

from .preprocessor import DataPreprocessor
from .embeddings import EmbeddingGenerator
from .vector_database import AdvancedVectorDatabase

# Suppress warnings
if hasattr(config, 'SUPPRESS_WARNINGS') and config.SUPPRESS_WARNINGS:
    warnings.filterwarnings('ignore')

class RAGRetriever:
    """Complete RAG Retrieval System"""
    
    def __init__(self, 
                 db_path: str = "rag_vector_database",
                 embedding_model: str = None,
                 language_model: str = None,
                 chunk_size: int = None,
                 top_k: int = None):
        
        # Use config.py values as defaults
        self.db_path = db_path
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.language_model = language_model or config.LANGUAGE_MODEL
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.top_k = top_k or config.TOP_K_RETRIEVAL
        self.ollama_url = config.OLLAMA_URL
        self.generation_options = config.GENERATION_OPTIONS
        
        # Initialize components
        print("🔧 Initializing RAG Retrieval System...")
        print(f"   🔗 Embedding model: {self.embedding_model}")
        print(f"   🤖 Language model: {self.language_model}")
        print(f"   📏 Chunk size: {self.chunk_size}")
        print(f"   🔍 Top-K retrieval: {self.top_k}")
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all RAG components"""
        try:
            # 1. Document Preprocessor
            self.preprocessor = DataPreprocessor(chunk_size=self.chunk_size)
            print("✅ Preprocessor initialized")
            
            # 2. Embedding Generator
            self.embedder = EmbeddingGenerator(model_name=self.embedding_model)
            print("✅ Embedding generator initialized")
            
            # 3. Vector Database
            self.vector_db = AdvancedVectorDatabase(self.db_path)
            print("✅ Vector database initialized")
            
            # 4. Test language model connection
            self._test_language_model()
            
            print("🎉 RAG system fully initialized!")
            
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            raise
    
    def _test_language_model(self):
        """Test connection to language model"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.language_model,
                    "prompt": "Test",
                    "stream": False,
                    "options": {"num_predict": 5}
                },
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ Language model connection verified")
            else:
                print(f"⚠️ Language model test failed: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Language model connection error: {e}")
    
    def add_documents_from_file(self, file_path: str, 
                               chunking_strategy: str = "lines") -> bool:
        """
        Add documents from file to the retrieval system
        
        Args:
            file_path: Path to the document file
            chunking_strategy: How to chunk the documents
            
        Returns:
            Success status
        """
        try:
            print(f"📚 Adding documents from {file_path}...")
            
            # Step 1: Preprocess documents
            chunks, stats = self.preprocessor.process_file(file_path, chunking_strategy)
            print(f"   📝 Created {len(chunks)} chunks")
            
            # Step 2: Generate embeddings
            embeddings = self.embedder.generate_batch_embeddings(chunks, show_progress=True)
            valid_embeddings = [emb for emb in embeddings if emb is not None]
            print(f"   🔗 Generated {len(valid_embeddings)} embeddings")
            
            # Step 3: Add to vector database
            metadata_list = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    'chunk_id': i,
                    'source_file': file_path,
                    'chunking_strategy': chunking_strategy,
                    'chunk_length': len(chunk),
                    'word_count': len(chunk.split())
                }
                metadata_list.append(metadata)
            
            self.vector_db.add_documents(chunks, embeddings, metadata_list)
            print(f"   ✅ Documents successfully added to database")
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, 
                                top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve (default: self.top_k)
            
        Returns:
            List of relevant chunks with metadata
        """
        if top_k is None:
            top_k = self.top_k
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.generate_embedding(query)
            
            if query_embedding is None:
                print("❌ Failed to generate query embedding")
                return []
            
            # Search in vector database
            results = self.vector_db.search(query_embedding, top_k=top_k)
            
            print(f"🔍 Retrieved {len(results)} relevant chunks for query: '{query[:50]}...'")
            
            return results
            
        except Exception as e:
            print(f"❌ Error retrieving chunks: {e}")
            return []
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate response using language model with retrieved context
        
        Args:
            query: User query
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Generated response
        """
        try:
            # Prepare context
            context_texts = [chunk['text'] for chunk in context_chunks]
            context = "\n\n".join(context_texts)
            
            # Create prompt
            prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response using Ollama with your config settings
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.language_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": self.generation_options
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                if answer:
                    print(f"🤖 Generated response ({len(answer)} characters)")
                    return answer
                else:
                    return "I couldn't generate a proper response based on the available information."
            else:
                print(f"❌ Language model API error: {response.status_code}")
                return "Sorry, I encountered an error while generating the response."
                
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "Sorry, I couldn't generate a response due to a technical error."
    
    def query(self, user_query: str, 
              top_k: Optional[int] = None,
              include_sources: bool = True) -> Dict[str, Any]:
        """
        Complete RAG query pipeline
        
        Args:
            user_query: User's question
            top_k: Number of chunks to retrieve
            include_sources: Whether to include source information
            
        Returns:
            Complete response with answer and metadata
        """
        start_time = time.time()
        
        print(f"\n🔍 Processing query: '{user_query}'")
        print("=" * 60)
        
        # Step 1: Retrieve relevant chunks
        print("1️⃣ Retrieving relevant information...")
        relevant_chunks = self.retrieve_relevant_chunks(user_query, top_k)
        
        if not relevant_chunks:
            return {
                'query': user_query,
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'retrieval_time': time.time() - start_time,
                'generation_time': 0,
                'total_time': time.time() - start_time
            }
        
        retrieval_time = time.time() - start_time
        
        # Step 2: Generate response
        print("2️⃣ Generating response...")
        generation_start = time.time()
        answer = self.generate_response(user_query, relevant_chunks)
        generation_time = time.time() - generation_start
        
        # Step 3: Prepare response
        sources = []
        if include_sources:
            for i, chunk in enumerate(relevant_chunks):
                source_info = {
                    'rank': i + 1,
                    'similarity': chunk['similarity'],
                    'text': chunk['text'],
                    'source_file': chunk['metadata'].get('source_file', 'unknown'),
                    'chunk_id': chunk['metadata'].get('chunk_id', i)
                }
                sources.append(source_info)
        
        total_time = time.time() - start_time
        
        response = {
            'query': user_query,
            'answer': answer,
            'sources': sources,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': total_time,
            'chunks_retrieved': len(relevant_chunks)
        }
        
        print(f"✅ Query completed in {total_time:.2f}s")
        return response
    
    def batch_query(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries"""
        results = []
        
        print(f"📋 Processing {len(queries)} queries...")
        
        for i, query in enumerate(queries):
            print(f"\n🔄 Query {i+1}/{len(queries)}")
            result = self.query(query)
            results.append(result)
        
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            db_stats = self.vector_db.get_statistics()
            embedder_stats = self.embedder.get_cache_stats()
            
            stats = {
                'database': db_stats,
                'embeddings': embedder_stats,
                'configuration': {
                    'embedding_model': self.embedding_model,
                    'language_model': self.language_model,
                    'chunk_size': self.chunk_size,
                    'top_k': self.top_k,
                    'database_path': self.db_path
                }
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting system stats: {e}")
            return {}

def interactive_rag_chat(rag_system: RAGRetriever):
    """Interactive chat interface for RAG system"""
    print("\n💬 Interactive RAG Chat")
    print("=" * 40)
    print("Type your questions about the documents!")
    print("Commands: 'quit' to exit, 'stats' for system info")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n🤔 Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'stats':
                stats = rag_system.get_system_stats()
                print(f"\n📊 System Statistics:")
                print(f"   📚 Documents: {stats['database']['total_documents']}")
                print(f"   🔗 Embedding cache: {stats['embeddings']['cache_size']}")
                print(f"   💾 Database size: {stats['database']['disk_usage_mb']:.2f} MB")
                continue
            elif not user_input:
                print("Please enter a question!")
                continue
            
            # Process query
            response = rag_system.query(user_input)
            
            # Display results
            print(f"\n🤖 Answer:")
            print(f"   {response['answer']}")
            
            print(f"\n📚 Sources ({len(response['sources'])} chunks):")
            for source in response['sources'][:2]:  # Show top 2 sources
                print(f"   {source['rank']}. Similarity: {source['similarity']:.3f}")
                print(f"      {source['text'][:80]}...")
            
            print(f"\n⏱️ Timing: Retrieval {response['retrieval_time']:.2f}s, "
                  f"Generation {response['generation_time']:.2f}s")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def test_complete_rag_system():
    """Test the complete RAG system"""
    print("🧪 Testing Complete RAG System")
    print("=" * 50)
    
    # Initialize RAG system
    rag = RAGRetriever()
    
    # Add documents (assuming cat-facts.txt exists)
    success = rag.add_documents_from_file("cat-facts.txt")
    
    if not success:
        print("❌ Failed to add documents. Make sure cat-facts.txt exists.")
        return None
    
    # Test queries
    test_queries = [
        "How fast can cats run?",
        "How much do cats sleep?",
        "What are cats' hearing abilities?",
        "How many toes do cats have?",
        "What is a group of cats called?"
    ]
    
    print(f"\n🧪 Testing with {len(test_queries)} queries...")
    
    for query in test_queries:
        print(f"\n" + "="*60)
        response = rag.query(query)
        
        print(f"❓ Question: {response['query']}")
        print(f"🤖 Answer: {response['answer']}")
        print(f"📊 Retrieved {response['chunks_retrieved']} chunks in {response['total_time']:.2f}s")
    
    print(f"\n✅ RAG system test completed!")
    return rag

if __name__ == "__main__":
    # Test the system
    rag_system = test_complete_rag_system()
    
    # Start interactive chat if test successful
    if rag_system:
        interactive_rag_chat(rag_system)